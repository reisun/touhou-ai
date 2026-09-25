"""Train/evaluate diagnostic policies with durable, inspectable run artifacts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import signal
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from touhou_ai.env import MockBridgeEnv


class StopRequested(Exception):
    pass


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"backend", "reward_profile", "seed", "total_timesteps", "n_steps",
                "batch_size", "n_epochs", "frames_per_step", "checkpoint_frequency",
                "evaluation_episodes"}
    if set(config) != required:
        raise ValueError("configuration keys must match configs/mock-smoke.json")
    if config["backend"] != "mock" or config["reward_profile"] != "mock_zero_test_only":
        raise ValueError("real-game training and rewards are not configured")
    limits = {"seed": (0, 2**31 - 1), "total_timesteps": (1, 1_000_000),
              "n_steps": (2, 8192), "batch_size": (2, 8192), "n_epochs": (1, 20),
              "frames_per_step": (1, 8), "checkpoint_frequency": (1, 1_000_000),
              "evaluation_episodes": (1, 100)}
    for key, (low, high) in limits.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"invalid {key}")
    if config["n_steps"] % config["batch_size"]:
        raise ValueError("n_steps must be divisible by batch_size")
    if config["total_timesteps"] % config["n_steps"]:
        raise ValueError("total_timesteps must be divisible by n_steps; prevents overshoot")
    return config


class RunCallback(BaseCallback):
    def __init__(self, output, frequency, stopping):
        super().__init__()
        self.output, self.frequency, self.stopping = output, frequency, stopping

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            self.model.save(self.output / f"checkpoint-{self.num_timesteps}.zip")
            write_json(self.output / "progress.json", {"timesteps": self.num_timesteps})
        return not self.stopping[0] and not (self.output / "STOP").exists()


def evaluate_model(model, env, episodes, seed, should_stop=lambda: False):
    results = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        reward_sum = 0.0
        for step in range(600):
            if should_stop():
                raise StopRequested("evaluation stopped")
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                results.append({"episode": episode, "return": reward_sum,
                                "steps": step + 1, "frame": info["frame"],
                                "terminated": terminated, "truncated": truncated})
                break
        else:
            raise RuntimeError("evaluation episode exceeded diagnostic limit")
    return {"backend": "mock", "reward_profile": "mock_zero_test_only", "episodes": results}


def train(config_path, output, resume=None):
    config = read_config(config_path)
    # Refuse overwriting completed runs; models are loaded only from local trusted runs.
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "starting", "started_at": time.time(), "backend": "mock"}
    write_json(output / "config.json", config)
    write_json(output / "status.json", status)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "gymnasium", "stable-baselines3", "numpy")}
    write_json(output / "versions.json", versions)
    stopping = [False]
    previous = {}
    for event in (signal.SIGTERM, signal.SIGINT):
        previous[event] = signal.signal(event, lambda *_: stopping.__setitem__(0, True))
    env = None
    model = None
    try:
        env = Monitor(MockBridgeEnv(frames_per_step=config["frames_per_step"]),
                      str(output / "episodes.monitor.csv"))
        if resume:
            prior = read_config(Path(resume).parent / "config.json")
            for key in ("backend", "reward_profile", "frames_per_step", "n_steps", "batch_size", "n_epochs"):
                if prior[key] != config[key]:
                    raise ValueError(f"resume configuration mismatch: {key}")
            model = PPO.load(resume, env=env, device="cpu")
        else:
            model = PPO("MlpPolicy", env, n_steps=config["n_steps"],
                        batch_size=config["batch_size"], n_epochs=config["n_epochs"],
                        seed=config["seed"], device="cpu", verbose=1)
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(str(output), ["stdout", "log", "csv"]))
        status["status"] = "running"
        write_json(output / "status.json", status)
        model.learn(total_timesteps=config["total_timesteps"], reset_num_timesteps=not bool(resume),
                    callback=RunCallback(output, config["checkpoint_frequency"], stopping))
        model.save(output / "model.zip")
        stopped = stopping[0] or (output / "STOP").exists()
        if not stopped:
            write_json(output / "evaluation.json", evaluate_model(
                model, env, config["evaluation_episodes"], config["seed"] + 1000,
                lambda: stopping[0] or (output / "STOP").exists()))
        status.update(status="stopped" if stopped else "completed", timesteps=model.num_timesteps)
    except StopRequested:
        status.update(status="stopped", timesteps=model.num_timesteps)
    except BaseException as error:
        status.update(status="failed", error=type(error).__name__)
        raise
    finally:
        if model is not None and hasattr(model, "_logger"):
            model.logger.close()
        if env is not None:
            env.close()
        for event, handler in previous.items():
            signal.signal(event, handler)
        status["finished_at"] = time.time()
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    training = commands.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        env = MockBridgeEnv()
        try:
            check_env(env, warn=True)
            print("PASS: Gymnasium/SB3 bridge environment contract")
        finally:
            env.close()
    elif args.command == "train":
        train(args.config, args.output, args.resume)
    else:
        if args.output.exists():
            raise FileExistsError("evaluation output already exists")
        config = read_config(args.model.parent / "config.json")
        env = MockBridgeEnv(frames_per_step=config["frames_per_step"])
        try:
            model = PPO.load(args.model, device="cpu")
            result = evaluate_model(model, env, config["evaluation_episodes"], config["seed"] + 1000)
            write_json(args.output, result)
            print(json.dumps(result))
        finally:
            env.close()


if __name__ == "__main__":
    main()

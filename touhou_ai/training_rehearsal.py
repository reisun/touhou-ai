"""Bounded synthetic PPO integration rehearsal, never an actual game adapter."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from touhou_ai.game_policy import NumericalEncoder, action_space, load_profile
from touhou_ai.game_rewards import EventRewards
from touhou_ai.policy_check import NumericalFeatures, synthetic_state


class RehearsalEnv(gym.Env):
    """Small action-dependent fixture with authoritative synthetic events."""
    def __init__(self, profile, horizon=32):
        self.encoder = NumericalEncoder(profile)
        self.observation_space = self.encoder.space
        self.action_space = action_space()
        self.rewards = EventRewards(profile["provisional_reward_weights"])
        self.horizon = horizon
        self.episode = 0
        self.finished = True
        self.history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.tick = 0
        self.finished = False
        self.total = 0
        self.state = synthetic_state()
        self.state["enemies"] = [{"id": "fixture-enemy", "x": 192, "y": 80,
            "width": 24, "height": 24, "vx": 0, "vy": 0, "hp": 100, "phase_initial_hp": 100}]
        self.rewards.reset(str(self.episode))
        return self.encoder.encode(self.state), {"source": "synthetic_rehearsal"}

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset required")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        self.tick += 1
        direction, shoot, focus, bomb = map(int, action)
        dx, dy = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1),
                  (0, 1), (-1, 1), (-1, 0), (-1, -1)][direction]
        player = self.state["player"]
        distance = 3 if focus else 6
        x, y = player["x"], player["y"]
        player["x"] = float(np.clip(x+dx*distance, 8, 376))
        player["y"] = float(np.clip(y+dy*distance, 16, 432))
        player["vx"], player["vy"] = (player["x"]-x)*30, (player["y"]-y)*30
        player["seconds_since_hit"] += 1/30
        player["seconds_since_bomb"] += 1/30
        events = []

        def event(kind, **fields):
            events.append({"id": f"{self.tick}:{kind}", "kind": kind,
                           "confirmed": True, **fields})

        activated = bool(bomb and player["bomb_available"])
        if activated:
            player["power"] = 0
            player["bomb_available"] = False
            player["seconds_since_bomb"] = 0
            event("bomb")
        enemies = self.state["enemies"]
        if shoot and enemies and abs(player["x"]-192) < 80:
            enemy = enemies[0]
            amount = min(10, enemy["hp"])
            enemy["hp"] -= amount
            event("damage", amount=amount, phase_initial_hp=100, bomb_active=activated)
            if enemy["hp"] == 0:
                event("kill", subject=enemy["id"])
                self.state["enemies"] = []
        self.state["bullets"] = [{"id": "fixture-bullet", "x": 192,
            "y": 300+self.tick*4, "vx": 0, "vy": 120, "ax": 0, "ay": 0,
            "radius_x": 3, "radius_y": 3}]
        if self.tick == 24 and abs(player["x"]-192) < 24:
            player["lives"] -= 1
            player["seconds_since_hit"] = 0
            event("hit")
        self.finished = self.tick >= self.horizon
        reward, components = self.rewards.calculate(str(self.episode), events)
        self.total += reward
        if self.finished:
            self.history.append({"episode": self.episode, "steps": self.tick,
                                 "return": self.total, "source": "synthetic_rehearsal"})
        return self.encoder.encode(self.state), reward, False, self.finished, {
            "source": "synthetic_rehearsal", "events": events, "reward_components": components}


def rehearse(profile_path, output, updates=2):
    if type(updates) is not int or not 1 <= updates <= 8:
        raise ValueError("updates must be 1..8")
    profile = load_profile(profile_path)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = RehearsalEnv(profile)
    settings = profile["provisional_ppo"] | {"n_steps": 32, "batch_size": 16, "n_epochs": 2}
    model = PPO("MultiInputPolicy", env, device="cpu", **settings,
                policy_kwargs={"features_extractor_class": NumericalFeatures,
                               "net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    result = {"status": "running", "source": "synthetic_rehearsal",
              "gameplay_training_steps": 0, "live_training_enabled": False}
    try:
        model.learn(total_timesteps=32*updates)
        changed = any(not torch.equal(initial[k], v) for k, v in model.policy.state_dict().items())
        if not changed or not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise AssertionError("PPO update did not produce finite changed parameters")
        model.save(output / "synthetic-checkpoint.zip")
        observation, _ = env.reset(seed=7)
        expected = model.predict(observation, deterministic=True)[0]
        resumed = PPO.load(output / "synthetic-checkpoint.zip", env=env, device="cpu")
        np.testing.assert_array_equal(expected, resumed.predict(observation, deterministic=True)[0])
        before = resumed.num_timesteps
        resumed.learn(total_timesteps=32, reset_num_timesteps=False)
        if resumed.num_timesteps != before+32:
            raise AssertionError("resume timestep mismatch")
        resumed.save(output / "synthetic-resumed.zip")
        result.update(status="passed", parameters_changed=changed, checkpoint_reload=True,
                      synthetic_training_steps=resumed.num_timesteps, resumed_steps=32,
                      completed_episodes=len(env.history), effective_ppo=settings)
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        env.close()
        (output / "episodes.json").write_text(json.dumps(env.history, indent=2)+"\n", encoding="utf-8")
        (output / "result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="/configs/sharu-inspired-v1.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    rehearse(args.profile, args.output, args.updates)

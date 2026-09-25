"""Bounded on-policy learning from the real game; staged, explicitly partial contract."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import configure

from touhou_ai.game_rewards import EventRewards
from touhou_ai.live_runtime import LiveRuntime, input_mask
from touhou_ai.live_reset import start_episode, continue_episode, wait_game_over, GameOverHold, advance_dialogue
from touhou_ai.policy_check import NumericalFeatures
from touhou_ai.telemetry import packet


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "th10-live-observed-v1"
CAPACITIES = {"bullets": 128, "enemies": 24, "items": 40, "lasers": 64}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01)


def publish_telemetry(data):
    # A read-only viewer cannot invalidate the learning trajectory if its file is busy.
    try:
        atomic_json(ROOT / "artifacts/telemetry/latest.json", data)
        return True
    except OSError:
        return False


class ObservedContract(gym.Env):
    """Only measured features. Availability masks distinguish absent managers.

    Deliberately separate from the full Sharu encoder: no invented HP, IDs,
    acceleration, elapsed-hit/bomb state or calibrated collision radii.
    """
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "player": gym.spaces.Box(-1, 1, (15,), dtype=np.float32),
            "far_grid": gym.spaces.Box(-1, 1, (3, 112, 96), dtype=np.float32),
            **{k: gym.spaces.Box(-1, 1, (n, 5), dtype=np.float32) for k, n in CAPACITIES.items()}})
        # Bomb is a one-choice head: disabling it must not corrupt PPO log probabilities.
        self.action_space = gym.spaces.MultiDiscrete([9, 2, 2, 1])

    def reset(self, **kwargs):
        raise RuntimeError("real collector owns reset")

    def step(self, action):
        raise RuntimeError("real collector owns stepping")

    def encode(self, raw):
        if raw["player"] is None:
            raise ValueError("player missing")
        if raw.get("lasers"):
            raise ValueError("live laser geometry has not passed acceptance")
        out = {k: np.zeros(s.shape, dtype=np.float32) for k, s in self.observation_space.spaces.items()}
        x, y = raw["player"]["position"]
        player = raw["player"]
        out["player"][:] = [x/192, y/448, player["velocity_raw"][0]/1000,
            player["velocity_raw"][1]/1000, raw["lives_raw"]/8, raw["power_raw"]/100,
            player["status"]/4, player["invincibility_raw"]/300, player["focus_raw"],
            *[raw.get(k) is not None for k in CAPACITIES], raw["stage"]/6, 1]
        for kind, capacity in CAPACITIES.items():
            entities = raw.get(kind)
            if entities is None:
                continue
            # No temporal identity is claimed; permutation-invariant pooling uses this frame only.
            ordered = sorted(entities, key=lambda e: ((e["position"][0]-x)**2+(e["position"][1]-y)**2,
                                                      *e["position"], *e["velocity_raw"]))
            for index, entity in enumerate(ordered):
                ex, ey = entity["position"]
                vx, vy = entity["velocity_raw"]
                if index < capacity:
                    out[kind][index] = [1, (ex-x)/384, (ey-y)/448, vx/10, vy/10]
                elif kind == "bullets" and 0 <= ex+192 < 384 and 0 <= ey < 448:
                    out["far_grid"][:, int(ey//4), int((ex+192)//4)] += [1, vx/10, vy/10]
        counts = out["far_grid"][0].copy()
        out["far_grid"][1:] /= np.maximum(counts, 1)
        out["far_grid"][0] = np.log1p(counts)/np.log(17)
        for array in out.values():
            if not np.isfinite(array).all():
                raise ValueError("nonfinite live observation")
            np.clip(array, -1, 1, out=array)
        return out


class ExtendedObservedContract(ObservedContract):
    def __init__(self):
        from touhou_ai.live_features import extended_space
        super().__init__()
        self.base = ObservedContract()
        self.observation_space = extended_space(self.base.observation_space)
        self.action_space = gym.spaces.MultiDiscrete([9, 2, 2, 2])

    def encode(self, raw):
        from touhou_ai.live_features import encode_extended
        base = self.base.encode(raw | {'lasers': []})
        # Preserve the first five channels used by the previous model for each laser.
        x, y = raw['player']['position']
        ordered = sorted(raw.get('lasers') or [], key=lambda e: ((e['position'][0]-x)**2+(e['position'][1]-y)**2,
                                                               *e['position'], *e['velocity_raw']))
        base['player'][12] = raw.get('lasers') is not None
        for index, entity in enumerate(ordered[:64]):
            ex, ey = entity['position']
            vx, vy = entity['velocity_raw']
            base['lasers'][index] = [1, (ex-x)/384, (ey-y)/448, vx/10, vy/10]
        return encode_extended(raw, base)


def hit_events(before, after):
    if (after["stage"] != before["stage"] or after["replay_mode"] != 0 or after["mode_flags"] != 0):
        raise ValueError("unvalidated stage/menu transition")
    difference = before["lives_raw"]-after["lives_raw"]
    if difference not in (0, 1):
        raise ValueError("unexpected life change; do not infer a reward")
    return ([{"id": f"life-loss:{after['stage_frame']}", "kind": "hit", "confirmed": True}]
            if difference else [])


def fingerprint(model):
    digest = hashlib.sha256()
    for parameter in model.policy.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def resume_manifest(checkpoint, extended=False):
    from touhou_ai.live_features import EXTENDED_CONTRACT
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("resume checkpoint must be a managed local artifact")
    manifest = json.loads((checkpoint.parent / "status.json").read_text(encoding="utf-8"))
    accepted = (CONTRACT, EXTENDED_CONTRACT) if extended else (CONTRACT,)
    if manifest.get("backend") != "real_th10" or manifest.get("contract") not in accepted:
        raise ValueError("not a compatible real-game checkpoint")
    matches = [e for e in manifest["episodes"] if e["checkpoint"] == checkpoint.name]
    if len(matches) != 1 or not matches[0].get("reload_verified"):
        raise ValueError("checkpoint is not verified by this run")
    expected = matches[0].get("checkpoint_sha256")
    if not expected or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("checkpoint integrity mismatch")
    return manifest, matches[0]


def migrate_extended(old, new):
    """Keep existing features/heads; new inputs start at zero influence. Optimizer resets."""
    source = old.policy.state_dict()
    target = new.policy.state_dict()
    for key, value in target.items():
        previous = source[key]
        if previous.shape == value.shape:
            value.copy_(previous)
        elif key.startswith('action_net.'):
            value.zero_()
            value[:13].copy_(previous[:13])
            if key.endswith('bias'):
                value[-1] = -4
        elif value.ndim == previous.ndim == 2 and value.shape[0] == previous.shape[0] and value.shape[1] > previous.shape[1]:
            value.zero_()
            value[:, :previous.shape[1]].copy_(previous)
        else:
            raise ValueError(f'unsupported migration: {key}')
    new.policy.load_state_dict(target)
    new.num_timesteps = old.num_timesteps
    new._n_updates = old._n_updates
    return new


def finish_buffer(buffer, count, last_values, terminal):
    if count < 2:
        raise ValueError("not enough real transitions to update")
    # PPO uses the collected episode, never zero-filled padding or invented transitions.
    buffer.observations = {k: v[:count] for k, v in buffer.observations.items()}
    for name in ("actions", "rewards", "episode_starts", "values", "log_probs", "advantages", "returns"):
        setattr(buffer, name, getattr(buffer, name)[:count])
    buffer.buffer_size = buffer.pos = count
    buffer.full = True
    buffer.compute_returns_and_advantage(last_values, np.asarray([terminal]))


def game_command(action):
    executable = shutil.which("pwsh")
    if not executable:
        raise RuntimeError("PowerShell 7 is required")
    result = subprocess.run([executable, "-NoProfile", "-File", str(ROOT / "scripts/game.ps1"), action],
                            cwd=ROOT, timeout=90, capture_output=True, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"game {action} failed: {result.stdout} {result.stderr}")
    print(result.stdout.strip(), flush=True)


def train(output, episodes=3, max_steps=1800, max_seconds=600, resume=None, continue_managed=False, extended=False, resume_paused=False):
    from touhou_ai.live_features import EXTENDED_CONTRACT, bomb_events
    contract_id = EXTENDED_CONTRACT if extended else CONTRACT
    if resume_paused and not continue_managed:
        raise ValueError('resume-paused requires continue-managed')
    if not 1 <= episodes <= 5 or not 32 <= max_steps <= 2400 or not 30 <= max_seconds <= 900:
        raise ValueError("bounded rehearsal budgets required")
    prior = resume_manifest(resume, extended) if resume is not None else None
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    contract = ExtendedObservedContract() if extended else ObservedContract()
    profile = json.loads((ROOT / "configs/sharu-inspired-v1.json").read_text(encoding="utf-8"))
    settings = profile["provisional_ppo"]
    model = PPO("MultiInputPolicy", contract, device="cpu", **settings,
                policy_kwargs={"features_extractor_class": NumericalFeatures,
                               "net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    if prior is not None:
        if prior[0]['contract'] != contract_id:
            model = migrate_extended(PPO.load(resume, device='cpu'), model)
        else:
            model = PPO.load(resume, env=contract, device="cpu")
        if model.num_timesteps != prior[1]["total_steps"]:
            raise ValueError("checkpoint timestep metadata mismatch")
        for key, value in settings.items():
            actual = getattr(model, key)
            if callable(actual):
                actual = actual(1.0)
            if actual != value:
                raise ValueError(f"resume PPO configuration differs: {key}")
    report = {"status": "running", "backend": "real_th10", "contract": contract_id,
              "episodes": [], "gameplay_training_steps": model.num_timesteps,
              "updates": prior[1]["total_updates"] if prior else 0,
              "full_profile_ready": False, "disabled": ["bomb", "damage_reward", "kill_reward",
              "stage_clear_reward", "stable_ids", "acceleration", "laser_geometry"],
              "configured_ppo": settings,
              "effective_ppo": settings | {"n_steps": "variable_episode_length"}, "max_seconds": max_seconds,
              "resumed_from": str(resume) if resume is not None else None,
              "progression": [{"stage": "input_and_update_connected", "enabled": ["Normal", "Reimu B",
                  "2 frames/action", "9 directions", "shoot", "focus", "hit=-5", "original PPO parameters"]}],
              "update_boundary": "game_over_menu", "telemetry_drops": 0,
              "between_episodes": "continue_in_same_process", "phase": "initializing",
              "first_episode_started_midplay": resume_paused}
    if extended:
        report['disabled'] = ['damage_reward', 'kill_reward', 'stage_clear_reward', 'stable_ids', 'laser_field_validation']
        report['extended_observations'] = ['bomb_state', 'enemy_hp_ratio', 'boss_flag', 'spell_id',
                                            'backward_acceleration', 'binary_derived_laser_rectangle']
        report['optimizer_reset_for_migration'] = bool(prior and prior[0]['contract'] != contract_id)
    atomic_json(output / "status.json", report)
    deadline = time.monotonic()+max_seconds
    initial_hash = fingerprint(model)
    runtime = hold = None
    try:
        for episode in range(episodes):
            if (output / "STOP").exists() or time.monotonic() >= deadline:
                report["status"] = "stopped"
                break
            budget = max_steps
            buffer = DictRolloutBuffer(budget, contract.observation_space, contract.action_space,
                                      device="cpu", gamma=settings["gamma"], gae_lambda=settings["gae_lambda"])
            rewards = EventRewards(profile["provisional_reward_weights"])
            episode_id = f"{output.name}-{episode+1}"
            rewards.reset(episode_id)
            total_reward, count, hits = 0.0, 0, 0
            terminal = False
            stop_requested = False
            log_path = output / f"episode-{episode+1}.jsonl"
            try:
                if runtime is None:
                    # An existing session is accepted only with an explicit switch and a verified game over.
                    if not continue_managed:
                        game_command("start")
                    record = json.loads((ROOT / ".runtime/game.json").read_text(encoding="utf-8-sig"))
                    runtime = LiveRuntime(record["Id"])
                    report["game_pid"] = record["Id"]
                    if continue_managed:
                        if resume_paused:
                            from touhou_ai.live_reset import resume_paused_episode
                            resume_paused_episode(runtime)
                        else:
                            continue_episode(runtime)
                    else:
                        start_episode(runtime)
                else:
                    continue_episode(runtime)
                report["phase"] = "playing"
                atomic_json(output / "status.json", report)
                before = runtime.snapshot()
                atomic_json(output / f"initial-state-{episode+1}.json", before)
                observation = contract.encode(before)
                with log_path.open("x", encoding="utf-8") as log:
                    for step in range(budget):
                        if (output / "STOP").exists() or time.monotonic() >= deadline:
                            stop_requested = True
                            break
                        if before.get('dialogue_raw'):
                            previous_lives = before['lives_raw']
                            before = advance_dialogue(runtime)
                            if before['lives_raw'] != previous_lives:
                                raise RuntimeError('life changed while advancing dialogue')
                            observation = contract.encode(before)
                        with torch.no_grad():
                            tensor, _ = model.policy.obs_to_tensor(observation)
                            action, values, log_prob = model.policy(tensor)
                            distributions = model.policy.get_distribution(tensor).distribution
                            probabilities = [d.probs[0].cpu().tolist() for d in distributions]
                        step_started = time.perf_counter()
                        chosen = action[0].cpu().numpy()
                        after = runtime.step_gameplay(input_mask(chosen.tolist()), 2)
                        after["sample_ms"] = (time.perf_counter()-step_started)*1000
                        if after["input_state_raw"][0] != input_mask(chosen.tolist()):
                            raise RuntimeError("actual input differs from sampled policy action")
                        events = hit_events(before, after)
                        if extended:
                            events += bomb_events(before, after)
                        reward, components = rewards.calculate(episode_id, events)
                        terminal = after["lives_raw"] < 0
                        if after["player"] is None and not terminal:
                            raise RuntimeError("player disappeared outside verified terminal")
                        next_observation = observation if after["player"] is None else contract.encode(after)
                        buffer.add(observation, chosen.reshape(1, -1), np.array([reward]),
                                   np.array([step == 0]), values, log_prob)
                        model.num_timesteps += 1
                        count += 1
                        hits += sum(event['kind'] == 'hit' for event in events)
                        total_reward += reward
                        telemetry = packet(after, episode_id, "live")
                        from touhou_ai.model_monitor import model_metadata
                        telemetry['model'] = model_metadata(model, contract_id,
                            str(resume) if resume else None,
                            reward_weights={k: profile['provisional_reward_weights'][k]
                                            for k in (['hit', 'bomb'] if extended else ['hit'])})
                        telemetry["policy"] = {"source": "real_th10", "trained_updates": report["updates"],
                            "value": float(values.item()), "action": chosen.tolist(),
                            "directions": probabilities[0], "shoot": probabilities[1][1],
                            "focus": probabilities[2][1], "bomb": probabilities[3][1] if extended else 0.0,
                            "observation_frame": before["stage_frame"]}
                        telemetry["reward"] = {"total": reward, "episode_return": total_reward,
                            "components": components, "enabled": ["hit", "bomb"] if extended else ["hit"]}
                        telemetry["learning"] = {"steps": model.num_timesteps, "updates": report["updates"],
                            "contract": contract_id, "partial_profile": True, "phase": "playing"}
                        telemetry["capabilities"].update(policy_connected=True, live_training=True)
                        log.write(json.dumps({"raw": after, "telemetry": telemetry, "events": events}, allow_nan=False)+"\n")
                        log.flush()
                        if not publish_telemetry(telemetry):
                            report["telemetry_drops"] += 1
                        observation, before = next_observation, after
                        if count % 120 == 0:
                            print(json.dumps({"episode": episode+1, "steps": count, "frame": after["stage_frame"],
                                              "hits": hits, "return": total_reward}), flush=True)
                        if terminal:
                            break
                with torch.no_grad():
                    last_values = model.policy.predict_values(model.policy.obs_to_tensor(observation)[0])
            finally:
                # Release the last policy action before any optimizer work.
                if runtime is not None and terminal:
                    wait_game_over(runtime)
            if stop_requested or not terminal:
                report["status"] = "stopped"
                report["stop_reason"] = "budget_or_stop_before_game_over; trajectory_not_updated"
                break
            report["phase"] = "optimizing_at_game_over"
            atomic_json(output / "status.json", report)
            def publish_hold(state):
                waiting = packet(state, episode_id, "live")
                waiting["learning"] = {"steps": model.num_timesteps, "updates": report["updates"],
                    "contract": contract_id, "partial_profile": True, "phase": "optimizing_at_game_over"}
                waiting["reward"] = {"total": 0, "episode_return": total_reward,
                    "components": {}, "enabled": ["hit"]}
                if not publish_telemetry(waiting):
                    report["telemetry_drops"] += 1
            hold = GameOverHold(runtime, publish_hold)
            hold.start()
            finish_buffer(buffer, count, last_values, terminal)
            model.rollout_buffer = buffer
            model.set_logger(configure(str(output / f"optimizer-{episode+1}"), ["json"]))
            prior_hash = fingerprint(model)
            model.train()
            if fingerprint(model) == prior_hash or not all(torch.isfinite(p).all() for p in model.policy.parameters()):
                raise AssertionError("real PPO update failed")
            model.logger.dump(model.num_timesteps)
            model.logger.close()
            checkpoint = output / f"real-episode-{episode+1}.zip"
            model.save(checkpoint)
            expected = model.predict(observation, deterministic=True)[0]
            old_steps = model.num_timesteps
            loaded = PPO.load(checkpoint, env=contract, device="cpu")
            np.testing.assert_array_equal(expected, loaded.predict(observation, deterministic=True)[0])
            if loaded.num_timesteps != old_steps or fingerprint(loaded) != fingerprint(model):
                raise AssertionError("checkpoint did not preserve real policy")
            model = loaded
            hold.stop()
            hold_samples = hold.samples
            hold = None
            report["updates"] += 1
            report["gameplay_training_steps"] = model.num_timesteps
            report["episodes"].append({"episode": episode+1, "steps": count, "hits": hits,
                "return": total_reward, "terminated": terminal, "truncated": not terminal,
                "final_frame": before["stage_frame"], "budget": budget, "checkpoint": checkpoint.name,
                "parameters_changed": True, "reload_verified": True,
                "game_pid": report["game_pid"], "optimizer_menu_samples": hold_samples,
                "updated_at_game_over": True,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "total_steps": model.num_timesteps, "total_updates": report["updates"]})
            if terminal and not any(p["stage"] == "game_over_boundary_observed" for p in report["progression"]):
                report["progression"].append({"stage": "game_over_boundary_observed", "episode": episode+1,
                    "evidence": "reserve lives -1, game-over menu held with neutral input throughout PPO/save/reload"})
            report["phase"] = "game_over_ready"
            final_telemetry = packet(runtime.snapshot(full=False), episode_id, "live")
            final_telemetry["learning"] = {"steps": model.num_timesteps, "updates": report["updates"],
                "contract": contract_id, "partial_profile": True, "phase": "game_over_ready"}
            publish_telemetry(final_telemetry)
            atomic_json(output / "status.json", report)
            print(json.dumps(report["episodes"][-1]), flush=True)
        else:
            report["status"] = "passed"
        report["initial_policy_sha256"] = initial_hash
        report["final_policy_sha256"] = fingerprint(model)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        try:
            if hold is not None:
                hold.stop()
        finally:
            if runtime is not None:
                try:
                    state = runtime.snapshot(full=False)
                    if (state.get('lives_raw', -1) >= 0 and state.get('mode_flags') == 0
                            and state.get('player') is not None and state.get('pause_words', [0, -1])[1] == 0):
                        from touhou_ai.live_acceptance import pause
                        report['paused_on_exit'] = pause(runtime)['pause_words'][1] == 2
                except Exception as error:
                    report['pause_on_exit_error'] = str(error)
                finally:
                    runtime.close()
            report["game_left_open"] = runtime is not None
            atomic_json(output / "status.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--continue-managed", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--resume-paused", action="store_true")
    args = parser.parse_args()
    try:
        train(args.output, args.episodes, args.max_steps, args.max_seconds, args.resume, args.continue_managed, args.extended, args.resume_paused)
    except BaseException as error:
        status_path = args.output / "status.json"
        if args.output.exists() and not status_path.exists():
            atomic_json(status_path, {"status": "failed", "backend": "real_th10", "contract": CONTRACT,
                "error": str(error), "episodes": [], "gameplay_training_steps": 0, "updates": 0,
                "resumed_from": str(args.resume) if args.resume is not None else None})
        raise

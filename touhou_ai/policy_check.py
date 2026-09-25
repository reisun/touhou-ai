"""Offline numerical-policy diagnostic. Does not collect or train on gameplay."""
import argparse
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from touhou_ai.game_policy import CAPACITIES, NumericalEncoder, action_space, load_profile


class NumericalFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.entities = nn.ModuleDict({"set_" + key: nn.Sequential(
            nn.Linear(observation_space[key].shape[1]-1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            for key in CAPACITIES})
        self.grid = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(128, 64), nn.ReLU())
        self.player = nn.Sequential(nn.Linear(observation_space['player'].shape[0], 32), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(352, 128), nn.ReLU())

    def forward(self, observations):
        parts = [self.player(observations["player"]), self.grid(observations["far_grid"])]
        for key in CAPACITIES:
            entity = observations[key]
            mask = entity[:, :, :1]
            encoded = self.entities["set_" + key](entity[:, :, 1:]) * mask
            parts.extend([encoded.sum(dim=1)/mask.sum(dim=1).clamp(min=1), encoded.max(dim=1).values])
        return self.merge(torch.cat(parts, dim=1))


def synthetic_state():
    return {"player": {"x": 192, "y": 400, "vx": 0, "vy": 0, "lives": 2, "power": 1,
                       "bomb_available": True, "seconds_since_hit": 10, "seconds_since_bomb": 10},
            "movement_bounds": {"left": 8, "top": 16, "right": 376, "bottom": 432},
            **{key: [] for key in CAPACITIES}}


class ContractOnlyEnv(gym.Env):
    def __init__(self, profile):
        self.observation_space = NumericalEncoder(profile).space
        self.action_space = action_space()

    def reset(self, **kwargs):
        raise RuntimeError("offline contract only; live reset/collector is not implemented")

    def step(self, action):
        raise RuntimeError("offline contract only; live input/collector is not implemented")


def check(profile_path, output):
    profile = load_profile(profile_path)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    encoder = NumericalEncoder(profile)
    state = synthetic_state()
    # Include every entity branch and far-field density in this synthetic fixture.
    state["bullets"] = [{"id": str(i), "x": i % 96 * 4, "y": i % 112 * 4,
                          "vx": 0, "vy": 120, "ax": 0, "ay": 0, "radius_x": 3, "radius_y": 3}
                         for i in range(160)]
    state["lasers"] = [{"id": "l", "x": 100, "y": 100, "angle_radians": 1, "length": 200, "width": 8}]
    state["enemies"] = [{"id": "e", "x": 192, "y": 60, "width": 24, "height": 24,
                         "vx": 0, "vy": 0, "hp": 50, "phase_initial_hp": 100}]
    state["items"] = [{"id": "i", "x": 180, "y": 240, "vx": 0, "vy": 60}]
    observation = encoder.encode(state)
    assert encoder.space.contains(observation)
    model = PPO("MultiInputPolicy", ContractOnlyEnv(profile), device="cpu",
                policy_kwargs={"features_extractor_class": NumericalFeatures,
                               "net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                **profile["provisional_ppo"])
    action, _ = model.predict(observation, deterministic=True)
    assert action_space().contains(action)
    tensor, _ = model.policy.obs_to_tensor(observation)
    features = model.policy.extract_features(tensor)
    features.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.policy.features_extractor.parameters() if p.grad is not None)
    model.policy.zero_grad()
    model.save(output / "untrained-policy.zip")
    loaded = PPO.load(output / "untrained-policy.zip", device="cpu")
    np.testing.assert_array_equal(action, loaded.predict(observation, deterministic=True)[0])
    from touhou_ai.telemetry import policy_packet
    (output / "policy-telemetry.json").write_text(json.dumps(policy_packet(loaded, observation), indent=2)+"\n", encoding="utf-8")
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        loaded.predict(observation, deterministic=True)
        timings.append((time.perf_counter()-start)*1000)
    result = {"status": "passed", "synthetic_only": True, "gameplay_training_steps": 0,
              "live_training_enabled": False, "parameters": sum(p.numel() for p in model.policy.parameters()),
              "prediction_ms_mean": float(np.mean(timings)), "prediction_ms_p95": float(np.percentile(timings, 95)),
              "note": "Model-only timing; does not establish end-to-end 30 Hz or gameplay ability."}
    (output / "profile.json").write_text(json.dumps(profile, indent=2)+"\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="/configs/sharu-inspired-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.profile, args.output)

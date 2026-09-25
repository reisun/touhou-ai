"""Numerical observation/action contract; deliberately contains no game input IO."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


CAPACITIES = {"bullets": 128, "lasers": 64, "enemies": 24, "items": 40}
FEATURES = {"bullets": 11, "lasers": 7, "enemies": 8, "items": 5}
DIRECTIONS = ((), ("up",), ("up", "right"), ("right",), ("down", "right"),
              ("down",), ("down", "left"), ("left",), ("up", "left"))


def load_profile(path):
    from touhou_ai.game_rewards import EventRewards
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile["profile"] != "sharu-inspired-v1" or profile["live_training_enabled"] is not False:
        raise ValueError("only disabled v1 live profiles are supported")
    if profile["game"] != {"difficulty": "Normal", "character": "Reimu", "shot": "B"}:
        raise ValueError("v1 requires Normal/Reimu B")
    if profile["observations"] != CAPACITIES | {"far_grid_cell_pixels": 4}:
        raise ValueError("unsupported observation layout")
    if profile["collection"] != {"game_fps": 60, "decisions_per_second": 30,
                                  "frames_per_action": 2, "instances": 1, "realtime": True,
                                  "demonstrations": False, "target_update_boundary": "game_over"}:
        raise ValueError("unsupported collection policy")
    scales = profile["local_encoding"]
    expected = {"width", "height", "velocity_scale", "acceleration_scale", "cpa_horizon_seconds",
                "elapsed_scale_seconds", "lives_scale", "power_scale"}
    if set(scales) != expected or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("invalid encoding scales")
    if scales["width"] != 384 or scales["height"] != 448:
        raise ValueError("v1 grid requires nominal 384x448 coordinates")
    EventRewards(profile["provisional_reward_weights"])
    return profile


def action_space():
    return gym.spaces.MultiDiscrete([9, 2, 2, 2])


def decode_action(action):
    array = np.asarray(action)
    if array.dtype.kind not in "iu" or not action_space().contains(array):
        raise ValueError("invalid action")
    direction, shoot, focus, bomb = map(int, array)
    return list(DIRECTIONS[direction]) + (["z"] if shoot else []) + (["shift"] if focus else []) + (["x"] if bomb else [])


class NumericalEncoder:
    def __init__(self, profile):
        self.scales = profile["local_encoding"]
        self.space = gym.spaces.Dict({
            "player": gym.spaces.Box(-1, 1, (15,), dtype=np.float32),
            "far_grid": gym.spaces.Box(-1, 1, (3, 112, 96), dtype=np.float32),
            **{key: gym.spaces.Box(-1, 1, (count, FEATURES[key]), dtype=np.float32)
               for key, count in CAPACITIES.items()}})

    @staticmethod
    def values(entity, *keys):
        values = np.asarray([entity[key] for key in keys], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite state")
        return values

    def encode(self, state):
        s, player = self.scales, state["player"]
        w, h, v, a = s["width"], s["height"], s["velocity_scale"], s["acceleration_scale"]
        x, y, vx, vy, lives, power, hit_time, bomb_time = self.values(
            player, "x", "y", "vx", "vy", "lives", "power", "seconds_since_hit", "seconds_since_bomb")
        left, top, right, bottom = self.values(state["movement_bounds"], "left", "top", "right", "bottom")
        if not (0 <= left < right <= w and 0 <= top < bottom <= h and left <= x <= right and top <= y <= bottom):
            raise ValueError("invalid calibrated movement bounds/player position")
        if min(lives, power, hit_time, bomb_time) < 0 or type(player["bomb_available"]) is not bool:
            raise ValueError("invalid player state")
        output = {key: np.zeros(space.shape, dtype=np.float32) for key, space in self.space.spaces.items()}
        output["player"][:] = [x/w, y/h, vx/v, vy/v, lives/s["lives_scale"], power/s["power_scale"],
                                (x-left)/w, (right-x)/w, (y-top)/h, (bottom-y)/h,
                                player["bomb_available"], hit_time/s["elapsed_scale_seconds"],
                                bomb_time/s["elapsed_scale_seconds"], (right-left)/w, (bottom-top)/h]
        for kind, capacity in CAPACITIES.items():
            entities = state[kind]
            ids = [entity["id"] for entity in entities]
            if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(set(ids)) != len(ids):
                raise ValueError("unique entity lifetime IDs required")
            for entity in entities:
                self.values(entity, "x", "y")
            ordered = sorted(entities, key=lambda e: ((e["x"]-x)**2+(e["y"]-y)**2, e["id"]))
            for index, entity in enumerate(ordered):
                ex, ey = self.values(entity, "x", "y")
                relative = [(ex-x)/w, (ey-y)/h]
                if kind == "bullets":
                    evx, evy, ax, ay, rx, ry = self.values(entity, "vx", "vy", "ax", "ay", "radius_x", "radius_y")
                    if min(rx, ry) < 0:
                        raise ValueError("negative bullet radius")
                    velocity = np.asarray([evx-vx, evy-vy])
                    delta = np.asarray([ex-x, ey-y])
                    speed2 = velocity @ velocity
                    t = np.clip(-(delta @ velocity)/speed2, 0, s["cpa_horizon_seconds"]) if speed2 else 0.0
                    distance = np.linalg.norm(delta + velocity*t)
                    features = [1, *relative, evx/v, evy/v, ax/a, ay/a, rx/w, ry/h,
                                distance/np.hypot(w, h), t/s["cpa_horizon_seconds"]]
                    if index >= capacity and 0 <= ex < w and 0 <= ey < h:
                        gx, gy = int(ex//4), int(ey//4)
                        output["far_grid"][:, gy, gx] += [1, evx/v, evy/v]
                elif kind == "lasers":
                    angle, length, width = self.values(entity, "angle_radians", "length", "width")
                    if min(length, width) < 0:
                        raise ValueError("negative laser geometry")
                    features = [1, *relative, np.sin(angle), np.cos(angle), length/h, width/w]
                elif kind == "enemies":
                    ew, eh, evx, evy, hp, maximum = self.values(entity, "width", "height", "vx", "vy", "hp", "phase_initial_hp")
                    if min(ew, eh, hp) < 0 or maximum <= 0 or hp > maximum:
                        raise ValueError("invalid enemy geometry/HP")
                    features = [1, *relative, ew/w, eh/h, evx/v, evy/v, hp/maximum]
                else:
                    evx, evy = self.values(entity, "vx", "vy")
                    features = [1, *relative, evx/v, evy/v]
                if index < capacity:
                    output[kind][index] = features
        counts = output["far_grid"][0].copy()
        output["far_grid"][1:] /= np.maximum(counts, 1)
        output["far_grid"][0] = np.log1p(counts)/np.log(17)
        for array in output.values():
            np.clip(array, -1, 1, out=array)
            if not np.isfinite(array).all():
                raise ValueError("observation overflow")
        return output

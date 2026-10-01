"""Versioned observer interface; unknown measurements are null, never invented."""
import math
import time
from touhou_ai.bullet_scope import display_observation


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point(entity):
    values = entity.get("position", [])
    if len(values) != 2 or not all(finite(v) for v in values):
        raise ValueError("invalid entity position")
    return [values[0]+192, values[1]]


def packet(raw, episode, source="recording", timestamp=None):
    if source not in {"recording", "live"}:
        raise ValueError("unknown telemetry source")
    player = raw.get("player")
    entities = {}
    for kind in ("bullets", "enemies", "items", "lasers", "player_shots"):
        values = raw.get(kind)
        entities[kind] = None if values is None else [
            {"position": point(entity), "velocity_raw": entity.get("velocity_raw"),
             "hitbox_raw": entity.get("hitbox_raw"), "type": entity.get("type"),
             "flags_raw": entity.get("flags_raw"),
             "hp": entity.get("hp"), "hp_max": entity.get("hp_max"),
             "is_boss": entity.get("is_boss"), "acceleration": entity.get("acceleration"),
             "age_frames": entity.get("age_frames"), "collision": entity.get("collision"),
             "geometry_raw": entity.get("angle_length_width_raw")}
            for entity in values]
    position = point(player) if player else None
    distances = [math.dist(position, b["position"]) for b in entities["bullets"] or []] if position else []
    frame = raw.get("stage_frame")
    return {"schema_version": 1, "episode_id": episode, "source": source,
            "ai_observation": display_observation(raw, entities),
            "timestamp": (time.time() if source == "live" else None) if timestamp is None else timestamp,
            "sequence": raw.get("gate_tick"), "game_frame": frame,
            "game": {"stage": raw.get("stage"), "difficulty": raw.get("difficulty"),
                     "character": raw.get("character"), "shot": raw.get("shot"),
                     "replay_mode": raw.get("replay_mode"), "mode_flags": raw.get("mode_flags"),
                     "lives_reserve": raw.get("lives_raw"),
                     "power": raw["power_raw"]/20 if finite(raw.get("power_raw")) else None},
            "player": None if player is None else {"position": position,
                "status": player.get("status"), "focus": player.get("focus_raw"),
                "velocity_raw": player.get("velocity_raw")},
            "entities": entities,
            "bomb": raw.get("bomb"), "spell": raw.get("spell"),
            "applied_input": raw.get("input_state_raw", [None])[0],
            "metrics": {"stage_seconds": frame/60 if finite(frame) else None,
                        "nearest_bullet_center_pixels": min(distances) if distances else None,
                        "sample_ms": raw.get("sample_ms")},
            "capabilities": {"frame_locked": raw.get("frame_locked") is True,
                             "stable_entity_ids": False,
                             "enemy_hp": any(e.get('hp') is not None for e in raw.get('enemies') or []),
                             "boss_phase": raw.get('spell') is not None,
                             "acceleration": any(e.get('acceleration') is not None for e in raw.get('bullets') or []),
                             "laser_geometry_available": any(e.get('collision') is not None for e in raw.get('lasers') or []),
                             "reward_events": raw.get('reward_events_verified') is True, "policy_connected": False,
                             "live_training": False,
                             "laser_geometry_validated": bool(raw.get('lasers')) and all(
                                 (e.get('collision') or {}).get('field_validated') is True
                                 for e in raw['lasers'])},
            "policy": None, "reward": None, "learning": None}


def policy_packet(model, observation):
    """Export actual SB3 heads for one diagnostic observation, not action samples."""
    import torch
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(observation)
        distribution = model.policy.get_distribution(tensor).distribution
        probabilities = [d.probs[0].cpu().tolist() for d in distribution]
        value = float(model.policy.predict_values(tensor).item())
    if list(map(len, probabilities)) != [9, 2, 2, 2]:
        raise ValueError("policy is not the game MultiDiscrete contract")
    return {"schema_version": 1, "source": "synthetic_diagnostic", "trained": False,
            "value": value, "directions": probabilities[0],
            "shoot": probabilities[1][1], "focus": probabilities[2][1], "bomb": probabilities[3][1],
            "direction_order": ["neutral", "up", "up_right", "right", "down_right",
                                "down", "down_left", "left", "up_left"]}

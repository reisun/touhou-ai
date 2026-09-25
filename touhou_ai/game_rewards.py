"""Confirmed-event rewards. No reward is inferred from missing entities or power."""
import math


KINDS = {"damage", "kill", "stage_clear", "hit", "bomb"}
DEFAULT_WEIGHTS = {"damage": 1.0, "kill": 0.5, "stage_clear": 20.0,
                   "remaining_life": 5.0, "hit": -5.0, "bomb": -0.5}


def number(value, name, low=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {name}")
    if not math.isfinite(value) or value < low:
        raise ValueError(f"invalid {name}")
    return value


class EventRewards:
    def __init__(self, weights=None):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if self.weights.keys() != DEFAULT_WEIGHTS.keys():
            raise ValueError("unexpected reward weights")
        for kind, value in self.weights.items():
            number(value, kind, -math.inf)
            if (kind in {"hit", "bomb"} and value >= 0) or (kind not in {"hit", "bomb"} and value <= 0):
                raise ValueError("reward signs must match the source policy")
        self.episode = None

    def reset(self, episode):
        if not isinstance(episode, str) or not episode:
            raise ValueError("episode ID required")
        self.episode, self.seen, self.unique = episode, {}, set()

    def calculate(self, episode, events):
        if self.episode is None or episode != self.episode:
            raise ValueError("reset required for this episode")
        # Validate and calculate on copies so rejected batches cannot consume events.
        seen, unique = dict(self.seen), set(self.unique)
        result = dict.fromkeys(KINDS, 0.0)
        for event in events:
            kind, event_id = event.get("kind"), event.get("id")
            if kind not in KINDS or not isinstance(event_id, str) or not event_id:
                raise ValueError("confirmed event kind and stable ID required")
            if event.get("confirmed") is not True:
                raise ValueError("unconfirmed game event")
            if event_id in seen:
                if event != seen[event_id]:
                    raise ValueError("conflicting duplicate event")
                continue
            reward = self.weights[kind]
            if kind == "damage":
                damage = number(event["amount"], "damage")
                hp = number(event["phase_initial_hp"], "phase_initial_hp")
                if hp == 0 or damage > hp or type(event.get("bomb_active")) is not bool:
                    raise ValueError("invalid damage event")
                reward *= 0 if event["bomb_active"] else damage / hp
            if kind in {"kill", "stage_clear"}:
                subject = event.get("subject")
                if not isinstance(subject, str) or not subject:
                    raise ValueError("stable lifetime/stage ID required")
                key = (kind, subject)
                if key in unique:
                    reward = 0.0
                elif kind == "stage_clear":
                    lives = event.get("remaining_lives")
                    if type(lives) is not int or lives < 0:
                        raise ValueError("invalid remaining lives")
                    reward += lives * self.weights["remaining_life"]
                unique.add(key)
            result[kind] += reward
            seen[event_id] = dict(event)
        self.seen, self.unique = seen, unique
        return sum(result.values()), result

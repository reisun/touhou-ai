"""Live launch gate. Configuration changes alone cannot certify an adapter."""
import json


def readiness():
    return {
        "schema_version": 1, "ready_for_live_training": False,
        "limited_live_training_implemented": True,
        "limited_contract": "th10-live-observed-v1",
        "note": "ready_for_live_training refers to the full Sharu profile, not the bounded hit-only rehearsal",
        "blockers": [
            "Validated raw-to-learner conversion, entity lifetimes and acceleration",
            "Authoritative enemy HP, phase identity and reward events",
            "Live laser geometry and calibrated movement bounds",
            "Multi-stage transitions and full-profile collector acceptance",
        ],
        "available": ["bounded live diagnostics", "numerical PPO synthetic rehearsal",
                      "real-game PPO rehearsal with observed inputs and life-loss reward"],
    }


def require_live_ready():
    report = readiness()
    if not report["ready_for_live_training"]:
        raise RuntimeError("Live training blocked: " + "; ".join(report["blockers"]))


if __name__ == "__main__":
    print(json.dumps(readiness(), indent=2))

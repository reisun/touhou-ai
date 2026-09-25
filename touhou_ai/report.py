"""Summarize persisted runs using only the standard library."""
import argparse
import json
from pathlib import Path


def summarize(root):
    results = []
    for path in sorted(root.glob("*/status.json")):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            results.append({"run": path.parent.name, "status": status["status"],
                            "timesteps": status.get("timesteps"),
                            "model_saved": (path.parent / "model.zip").exists(),
                            "evaluation_saved": (path.parent / "evaluation.json").exists()})
        except (ValueError, KeyError, OSError) as error:
            results.append({"run": path.parent.name, "status": "unreadable",
                            "error": type(error).__name__})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))

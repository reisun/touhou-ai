"""Bounded game-over menu inspection for the pinned, already managed process."""
import argparse
import json
from pathlib import Path
import struct

from touhou_ai.live_runtime import LiveRuntime
from touhou_ai.live_reset import start_episode


def inspect(runtime):
    state = runtime.snapshot(full=False)
    pointers = {}
    for address in range(0x4776e0, 0x477860, 4):
        value = runtime.reader.integer(address)
        if 0x10000 <= value < 0x70000000:
            try:
                pointers[hex(address)] = {"pointer": hex(value),
                    "words": list(struct.unpack("<32i", runtime.reader.block(value, 128)))}
            except (OSError, ValueError):
                continue
    return {"state": state, "pointers": pointers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--tap", type=lambda value: int(value, 0))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(Path('.runtime/game.json').read_text(encoding='utf-8-sig'))
    runtime = LiveRuntime(record['Id'])
    rows = []
    try:
        if args.start:
            start_episode(runtime)
            for i in range(1600):
                state = runtime.step_gameplay(0, 2, full=False)
                if state['lives_raw'] < 0:
                    break
            else:
                raise TimeoutError('no terminal reached within diagnostic budget')
            rows.append(inspect(runtime))
        if args.tap is not None:
            runtime.step(args.tap, 1, full=False)
            runtime.step(0, 1, full=False)
        for _ in range(8):
            runtime.step(0, 30, full=False)
            rows.append(inspect(runtime))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(rows, stream, indent=2)
        print(json.dumps(rows[-1]['state']))
    finally:
        runtime.close()

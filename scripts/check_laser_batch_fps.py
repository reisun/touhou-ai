"""Bounded passive post-deployment FPS check using existing shared telemetry."""
import json
import pathlib
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.telemetry_memory import read

root = pathlib.Path(__file__).resolve().parents[1]
record = json.loads((root / '.runtime/live-learning.json').read_text())
previous = None
rows = []
start = time.monotonic()
while time.monotonic() - start < 240:
    item = read(previous=previous)
    if item:
        previous, payload = item
        t = json.loads(payload); learning = t.get('learning') or {}; game = t.get('game') or {}
        if learning.get('phase') == 'playing' and t.get('episode_id', '').startswith(record['RunId']):
            entities = t.get('entities') or {}; frame = t.get('game_frame', 0)
            phase = ('boss' if game.get('stage') == 1 and frame >= 4000 and
                     any(e.get('is_boss') for e in entities.get('enemies', [])) else
                     'road' if game.get('stage') == 1 and frame < 2400 else 'other')
            rows.append(dict(time=t['timestamp'], episode=t['episode_id'], stage=game.get('stage'),
                             frame=frame, steps=learning['steps'], phase=phase))
    time.sleep(.05)
groups = {}
for a, b in zip(rows, rows[1:]):
    dt = b['time'] - a['time']; ds = b['steps'] - a['steps']
    if (a['episode'] == b['episode'] and a['stage'] == b['stage'] and a['phase'] == b['phase']
            and 0 < dt < 1 and 0 < ds <= 10 and b['frame'] - a['frame'] == ds * 2):
        entry = groups.setdefault(b['phase'], dict(seconds=0., frames=0, intervals=0))
        entry['seconds'] += dt; entry['frames'] += ds * 2; entry['intervals'] += 1
for entry in groups.values():
    entry['effective_fps'] = entry['frames'] / entry['seconds']
report = dict(run=record, samples=len(rows), groups=groups, rows=rows)
name = sys.argv[1] if len(sys.argv) > 1 else 'laser-batch-live.json'
if pathlib.Path(name).name != name or not name.endswith('.json'):
    raise ValueError('expected a JSON filename')
(root / 'artifacts/fps-diagnosis-20260930' / name).write_text(json.dumps(report))
print(json.dumps(groups), flush=True)

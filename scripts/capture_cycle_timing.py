"""Bounded read-only shared-memory timing sampler; one output file at exit."""
import json
import pathlib
import statistics
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.telemetry_memory import read

root = pathlib.Path(__file__).resolve().parents[1]
record = json.loads((root / '.runtime/live-learning.json').read_text())
previous = None; seen = set(); rows = []; start = time.monotonic(); laser_samples = 0
while time.monotonic() - start < 240:
    item = read(previous=previous)
    if item:
        previous, payload = item; packet = json.loads(payload)
        episode = packet.get('episode_id', '')
        timing = packet.get('previous_cycle_timing')
        if episode.startswith(record['RunId']) and timing:
            key = (episode, timing['step'])
            if key not in seen:
                seen.add(key); rows.append(dict(timing, episode=episode))
                if timing['active_lasers'] and timing['stage'] == 1:
                    laser_samples += 1
    if len(sys.argv) > 1 and laser_samples >= 500:
        break
    time.sleep(.025)
groups = {}; offsets = {}
for row in rows:
    key = (row['episode'], row['stage'])
    offset = row['frame'] - 2*row['step']
    bridged = key in offsets and offsets[key] != offset
    offsets[key] = offset
    if bridged or row['terminal'] or row['transition'] != 'gameplay':
        continue
    phase = ('boss_laser' if row['active_lasers'] else 'boss_no_laser') if row['stage'] == 1 and row['frame'] >= 4000 and row['boss'] else 'other'
    groups.setdefault(phase, []).append(row)
summary = {}
for phase, samples in groups.items():
    keys = [key for key in samples[0] if key.endswith('_ms')]
    summary[phase] = dict(samples=len(samples), mean={k: statistics.mean(r[k] for r in samples) for k in keys},
                         median={k: statistics.median(r[k] for r in samples) for k in keys},
                         active_lasers=statistics.mean(r['active_lasers'] for r in samples))
out = root / 'artifacts/fps-diagnosis-20260930' / ('cycle-timing-lasers.json' if len(sys.argv) > 1 else 'cycle-timing-live.json')
out.write_text(json.dumps(dict(run=record, summary=summary, rows=rows), indent=2))
print(json.dumps(summary), flush=True)

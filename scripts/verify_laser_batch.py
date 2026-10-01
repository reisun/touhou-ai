"""Replay captured observations against scalar SAT, and benchmark both encoders."""
import json
import pathlib
import statistics
import sys
import time
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from tests.test_laser_batch import reference
from touhou_ai.live_action_grid import LiveActionGridContract, live_risk_grid

root = pathlib.Path(__file__).resolve().parents[1]
out = root / 'artifacts/fps-diagnosis-20260930'
rows = json.loads((out / 'raws.json').read_text())
env = LiveActionGridContract()

def measure(fn):
    times = []
    for _ in range(9):
        start = time.perf_counter(); fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)

results = []
for row in rows:
    raw = row['raw']
    new = env.encode(raw)
    with patch('touhou_ai.live_action_grid.live_risk_grid', reference):
        old = env.encode(raw)
    for key in old:
        np.testing.assert_array_equal(new[key], old[key])
    if not any(x['collision']['active'] for x in raw['lasers']):
        continue
    result = dict(frame=raw['stage_frame'], old_risk_ms=measure(lambda: reference(raw)),
                  new_risk_ms=measure(lambda: live_risk_grid(raw)),
                  new_encode_ms=measure(lambda: env.encode(raw)))
    with patch('touhou_ai.live_action_grid.live_risk_grid', reference):
        result['old_encode_ms'] = measure(lambda: env.encode(raw))
    results.append(result)
summary = {key: statistics.median(x[key] for x in results)
           for key in ('old_risk_ms', 'new_risk_ms', 'old_encode_ms', 'new_encode_ms')}
report = dict(matching_observations=len(rows), active_laser_observations=len(results),
              medians=summary, scenes=results)
(out / 'laser-batch-verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)

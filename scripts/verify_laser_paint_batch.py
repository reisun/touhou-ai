"""Compare whole observations and paired CPU timings for laser rasterization."""
import json
import pathlib
import statistics
import sys
import time
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from tests.test_laser_paint_batch import reference
from touhou_ai.live_action_grid import LiveActionGridContract
from touhou_ai.dual_grid import paint_laser

root = pathlib.Path(__file__).resolve().parents[1]
out = root / 'artifacts/fps-diagnosis-20260930'
rows = json.loads((out / 'raws.json').read_text()); env = LiveActionGridContract()
results = []
for row in rows:
    raw = row['raw']; new = env.encode(raw)
    with patch('touhou_ai.dual_grid.paint_laser', reference):
        old = env.encode(raw)
    for key in old:
        np.testing.assert_array_equal(new[key], old[key])
    if not any(x['collision']['active'] for x in raw['lasers']):
        continue
    times = {'old': [], 'new': []}
    for i in range(11):
        for name, fn in ((('old', reference), ('new', paint_laser)) if i % 2 else
                         (('new', paint_laser), ('old', reference))):
            with patch('touhou_ai.dual_grid.paint_laser', fn):
                start = time.perf_counter(); env.encode(raw)
                times[name].append((time.perf_counter() - start)*1000)
    results.append(dict(frame=raw['stage_frame'], **{k: statistics.median(v) for k, v in times.items()}))
report = dict(matching_observations=len(rows), scenes=results,
              median_encode_ms={k: statistics.median(r[k] for r in results) for k in ('old', 'new')})
(out / 'laser-paint-verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)

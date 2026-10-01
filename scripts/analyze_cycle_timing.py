"""Exclude dialogue/stage bridges by the actual frame/decision offset."""
import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parents[1] / 'artifacts/fps-diagnosis-20260930'
groups = {}; seen = set(); excluded = 0
for name in ('cycle-timing-live.json', 'cycle-timing-lasers.json'):
    offsets = {}
    for row in json.loads((root/name).read_text())['rows']:
        key = (row['episode'], row['stage'])
        offset = row['frame'] - 2*row['step']
        bridged = key in offsets and offsets[key] != offset
        offsets[key] = offset
        if bridged or row['terminal'] or row['transition'] != 'gameplay':
            excluded += 1
            continue
        identity = (row['episode'], row['step'])
        if identity in seen:
            continue
        seen.add(identity)
        phase = ('boss_laser' if row['active_lasers'] else 'boss_no_laser') if row['stage'] == 1 and row['frame'] >= 4000 and row['boss'] else 'other'
        groups.setdefault(phase, []).append(row)
summary = {}
for phase, rows in groups.items():
    keys = [k for k in rows[0] if k.endswith('_ms')]
    summary[phase] = dict(samples=len(rows),
        mean={k: statistics.mean(r[k] for r in rows) for k in keys},
        median={k: statistics.median(r[k] for r in rows) for k in keys},
        bullets=statistics.mean(r['bullets'] for r in rows),
        active_lasers=statistics.mean(r['active_lasers'] for r in rows),
        over_70ms=sum(r['total_ms']>70 for r in rows))
report = dict(excluded=excluded, groups=summary)
(root/'cycle-timing-analysis.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)

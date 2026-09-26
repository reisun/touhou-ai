"""Read-only analysis of shot/bomb probabilities and fixed-scene checkpoints."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO
from touhou_ai.dual_grid import DualGridContract

torch.set_num_threads(1)
out = ROOT / 'artifacts/action-coupling'
out.mkdir(exist_ok=True)
records, checkpoints, probes = [], [], []
for path in sorted((ROOT / 'artifacts').glob('live-learning-*/status.json')):
    status = json.loads(path.read_text(encoding='utf-8-sig'))
    dual = status.get('contract') == 'th10-dual-grid-v1'
    old = status.get('contract') == 'th10-focused-bullets-v1'
    if not (dual or old):
        continue
    episodes = status.get('episodes', [])
    if old:
        episodes = [e for e in episodes if e['episode'] in [1, 2, 3, 50, 100, 150, 200, 250, 300, 350, 377, 378, 379, 380, 381]]
    for e in episodes:
        file = path.parent / f"episode-{e['episode']}.jsonl"
        if not file.exists():
            continue
        values, previous = [], None
        changes = []
        with file.open(encoding='utf-8') as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                t = row['telemetry']; p = t.get('policy'); r = t.get('reward', {}).get('components', {})
                if not p or 'shoot' not in p:
                    continue
                invalid = r.get('invalid_bomb', 0) < 0
                values.append([p['shoot'], p['bomb'], invalid, row['raw']['power_raw'] < 20,
                               p['action'][1], r.get('invalid_bomb', 0), row['raw']['stage_frame'] <= 480])
                if previous is not None:
                    changes.append([previous[0], p['shoot'] - previous[1], p['bomb'] - previous[2]])
                previous = (invalid, p['shoot'], p['bomb'])
                if path.parent.name == 'live-learning-20260926-204340-262e15' and e['episode'] == 1 and i % 27 == 0:
                    probes.append((row['raw'], r))
        a = np.asarray(values); ch = np.asarray(changes)
        def avg(mask, col):
            return float(a[mask, col].mean()) if mask.any() else None
        rec = {'run': path.parent.name, 'episode': e['episode'], 'architecture': status['contract'],
               'policy_updates': e['total_updates'] - 1, 'n': len(a),
               'shot': float(a[:, 0].mean()), 'bomb': float(a[:, 1].mean()),
               'invalid_fraction': float(a[:, 2].mean()), 'invalid_penalty': float(a[:, 5].sum()),
               'shot_early': avg(a[:, 6] > 0, 0), 'bomb_early': avg(a[:, 6] > 0, 1),
               'shot_low_power': avg(a[:, 3] > 0, 0), 'bomb_low_power': avg(a[:, 3] > 0, 1),
               'shot_on_invalid': avg(a[:, 2] > 0, 0),
               'next_shot_delta_after_invalid': float(ch[ch[:, 0] > 0, 1].mean()) if (ch[:, 0] > 0).any() else None,
               'return': e['return']}
        records.append(rec)
        if dual:
            checkpoints.append((e['total_updates'], path.parent / e['checkpoint']))
        print(json.dumps(rec), flush=True)

contract = DualGridContract()
neutral, penalized = [], []
for raw, rewards in probes:
    neutral.append(contract.encode(raw, {**rewards, 'invalid_bomb': 0.}))
    penalized.append(contract.encode(raw, {**rewards, 'invalid_bomb': -7 / 30}))
batch = {k: torch.tensor(np.stack([o[k] for o in neutral + penalized])) for k in neutral[0]}
fixed = []
for count, checkpoint in checkpoints:
    model = PPO.load(checkpoint, device='cpu', n_steps=64)
    with torch.no_grad():
        dist = model.policy.get_distribution(batch).distribution
        shot, bomb = [dist[i].probs[:, 1].numpy() for i in [1, 3]]
    n = len(neutral)
    fixed.append({'updates': count, 'checkpoint': str(checkpoint), 'scenes': n,
                  'shot': float(shot[:n].mean()), 'bomb': float(bomb[:n].mean()),
                  'shot_penalty_input_delta': float((shot[n:] - shot[:n]).mean()),
                  'bomb_penalty_input_delta': float((bomb[n:] - bomb[:n]).mean())})
    print('FIXED', json.dumps(fixed[-1]), flush=True)
    del model
result = {'episodes': records, 'fixed_scenes': fixed,
          'scope': 'all completed dual-grid episodes at startup; 15 selected old focused episodes; fixed scenes from one recorded episode',
          'fixed_probe': 'same raw scenes and other reward inputs; invalid_bomb input toggled between 0 and -7/30; checkpoint after update'}
(out / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

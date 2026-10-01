"""Read-only live telemetry/checkpoint verification; never sends game input."""
import sys, pathlib, json, time, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from touhou_ai.telemetry_memory import read
from touhou_ai.separate_clip_ppo import SeparateClipPPO

root = pathlib.Path('.')
run = json.loads((root / '.runtime/live-learning.json').read_text())
out = root / 'artifacts/live-action-grid-cutover-20260929'
samples = []
previous = None
deadline = time.monotonic() + 30
while len(samples) < 32 and time.monotonic() < deadline:
    item = read(previous=previous)
    if item:
        previous, payload = item
        d = json.loads(payload)
        if d.get('episode_id', '').startswith(run['RunId']) and d.get('learning', {}).get('phase') == 'playing':
            a = np.asarray(d['ai_observation']['action_grid'])
            assert a.shape == (4, 3, 3) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
            assert d['ai_observation']['action_grid_frame'] == d['policy']['observation_frame']
            assert d['policy']['action'][1] == d['policy']['action'][3] == 0
            assert d['reward']['enabled'] == ['hit']
            assert all(v == 0 for k, v in d['reward']['components'].items() if k != 'hit')
            samples.append(dict(frame=d['game_frame'], episode=d['episode_id'], action=d['policy']['action'], risk_cells=int(a.sum())))
    time.sleep(.07)
assert len(samples) == 32
s = json.loads((pathlib.Path(run['Output']) / 'status.json').read_text())
assert s['status'] == 'running' and s['resumed_from'] is None
e = s['episodes'][0]
assert e['reload_verified'] and e['parameters_changed'] and e['optimization']['backend'] == 'directml'
p = pathlib.Path(run['Output']) / e['checkpoint']
assert hashlib.sha256(p.read_bytes()).hexdigest() == e['checkpoint_sha256']
torch.set_num_threads(1)
m = SeparateClipPPO.load(p, device='cpu')
assert m.observation_space['action_grid'].shape == (4, 3, 3)
assert m.policy.pi_features_extractor is not m.policy.vf_features_extractor
extra = float(m.policy.pi_features_extractor.merge[0].weight[:, 321:].detach().abs().sum())
assert extra > 0
result = dict(run=run, status=s['status'], updates=s['updates'], steps=s['gameplay_training_steps'],
              fresh=s['resumed_from'] is None, first_episode=e, extra_actor_weight_abs_sum=extra,
              parameters=sum(p.numel() for p in m.policy.parameters()), samples=samples)
(out / 'live-action-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('samples','first_episode')}, indent=2))

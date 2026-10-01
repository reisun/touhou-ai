"""Check complete experiment records without controlling live training."""
import sys,pathlib,json,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.separate_clip_ppo import SeparateClipPPO
root=pathlib.Path('artifacts/risk-refinements-20260929')
torch.set_num_threads(1)
physics=hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest()
records=[];episodes=0
for variant in ('control','urgency','continuous','scale'):
    for seed in (7,17,27):
        folder=root/variant/str(seed)
        manifest=json.loads((folder/'manifest.json').read_text())
        result=json.loads((folder/'result.json').read_text())
        assert manifest['physics_sha256']==physics
        assert result['save_reload_equal']
        for mode in ('sample','greedy','frozen_sample','frozen_greedy'):
            evaluation=result['evaluations'][mode];rows=evaluation['episodes']
            assert len(rows)==96 and sorted(r['seed'] for r in rows)==list(range(5000,5096))
            assert np.isclose(evaluation['survival'],np.mean([r['success'] for r in rows]))
            assert np.isclose(evaluation['mean_frames'],np.mean([r['frames'] for r in rows]))
            assert all(0<r['frames']<=600 for r in rows)
            episodes+=len(rows)
        path=folder/'model.zip';m=SeparateClipPPO.load(path,device='cpu')
        assert m.num_timesteps==16384 and m.observation_space['action_grid'].shape==(4,3,3)
        assert sum(p.numel() for p in m.policy.parameters())==710472
        records.append(dict(variant=variant,seed=seed,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
out=dict(models=len(records),evaluation_episodes=episodes,physics_sha256=physics,records=records)
(root/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(models=len(records),evaluation_episodes=episodes,verified=True)))

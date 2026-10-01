"""Offline interventions on live checkpoint; no live weights are modified."""
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.checkpoint_rng import load_preserving_rng
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.live_action_grid import LiveActionGridContract
from touhou_ai.critic_normalization import install,VARIANTS
ROOT=pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
study=json.loads((ROOT/'artifacts/shot-reward-study-20260930/study.json').read_text())
capture=json.loads((ROOT/'artifacts/boss-reward-tanh-20260930/capture.json').read_text())
env=LiveActionGridContract();observations=[]
for row in capture['observations']:
    obs=env.encode(row['raw'],row['rewards'])
    if row['bomb_clock'] is not None:
        obs['bomb_clock']=np.asarray([((round(row['bomb_clock'][0]*12)+2)%12)/12],np.float32)
    observations.append(obs)
report={};base_probs=None;base_values=None
for variant in VARIANTS:
    model=load_preserving_rng(SeparateClipPPO,pathlib.Path(study['source']),device='cpu')
    install(model.policy,variant);model.policy.set_training_mode(False)
    captured=[];values=[];probs=[]
    h=model.policy.mlp_extractor.value_net[-1].register_forward_hook(lambda m,a,y:captured.append(y.detach().numpy()))
    with torch.no_grad():
        for o in observations:
            _,v,_,ds=model.policy.forward_with_distribution(model.policy.obs_to_tensor(o)[0])
            values.append(v.item());probs.append(np.concatenate([d.probs[0].numpy() for d in ds]))
    h.remove();a=np.concatenate(captured);values=np.asarray(values);probs=np.asarray(probs)
    if base_probs is None:base_probs=probs;base_values=values
    assert np.array_equal(probs,base_probs)
    report[variant]=dict(saturation=float((abs(a)>.99).mean()),value_std=float(values.std()),
        value_min=float(values.min()),value_max=float(values.max()),
        mean_abs_value_shift=float(abs(values-base_values).mean()),
        fixed_saturated_units=int((np.all(a>.99,0)|np.all(a<-.99,0)).sum()),
        actor_probabilities_identical=True,
        per_unit_scene_std_rms=float(np.sqrt(np.mean(a.std(0)**2))))
out=ROOT/'artifacts/critic-normalization-20260930'
(out/'live-probe.json').write_text(json.dumps(dict(source=study['source'],samples=len(observations),
    results=report,limitation='Immediate intervention only; values have no measured accuracy target. No retraining or live deployment.'),indent=2))
print(json.dumps(report,indent=2))

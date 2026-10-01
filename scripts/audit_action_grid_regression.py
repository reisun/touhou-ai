"""Check neutral extension outputs and checkpoint RNG continuity, no game input."""
import sys,pathlib,json,tempfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.live_action_grid import build_live_action_model,LiveActionGridContract
from touhou_ai.dual_grid import DualGridContract,GridRolloutBuffer
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.simulation_speed import distributions

torch.set_num_threads(1)
profile=json.loads(pathlib.Path('configs/sharu-inspired-v1.json').read_text())
settings=profile['provisional_ppo']|profile['evasion_ppo_overrides']
old=SeparateClipPPO(EvasionPolicy,DualGridContract(),device='cpu',rollout_buffer_class=GridRolloutBuffer,**settings,
    policy_kwargs=dict(share_features_extractor=False,features_extractor_class=NarrowGridFeatures,net_arch=dict(pi=[256,128],vf=[256,128])))
oldrng=torch.get_rng_state().clone();oldnrng=np.random.get_state()
new=build_live_action_model(settings)
assert torch.equal(oldrng,torch.get_rng_state())
raws=json.loads(pathlib.Path('artifacts/live-action-diagnosis-20260929/raws.json').read_text())
c=LiveActionGridContract();obs=[c.encode(r) for r in raws[:32]]
batch={k:np.stack([o[k] for o in obs]) for k in obs[0]}
base={k:v for k,v in batch.items() if k!='action_grid'}
with torch.no_grad():
    a=distributions(old.policy,base);b=distributions(new.policy,batch)
    delta=max(float((x.probs-y.probs).abs().max()) for x,y in zip(a,b))
    v0=old.policy.predict_values(old.policy.obs_to_tensor(base)[0]);v1=new.policy.predict_values(new.policy.obs_to_tensor(batch)[0])
    vd=float((v0-v1).abs().max())
print('initial parity',delta,vd,flush=True)
result=dict(initial_probability_max_error=delta,initial_value_max_error=vd,initial_rng_equal=True)
with tempfile.TemporaryDirectory() as tmp:
    new.save(pathlib.Path(tmp)/'model')
    torch.manual_seed(987);np.random.seed(987)
    expected_t=torch.get_rng_state().clone();expected_n=np.random.get_state()
    loaded=SeparateClipPPO.load(pathlib.Path(tmp)/'model.zip',device='cpu')
    result['load_changes_torch_rng']=not torch.equal(expected_t,torch.get_rng_state())
    result['load_changes_numpy_rng']=not np.array_equal(expected_n[1],np.random.get_state()[1])
    torch.set_rng_state(expected_t)
    before=[new.predict(obs[0],deterministic=False)[0].tolist() for _ in range(20)]
    torch.set_rng_state(expected_t)
    loaded=SeparateClipPPO.load(pathlib.Path(tmp)/'model.zip',device='cpu')
    after=[loaded.predict(obs[0],deterministic=False)[0].tolist() for _ in range(20)]
    result['identical_model_action_sequence_changed_by_reload']=before!=after
    result['before_actions']=before;result['after_actions']=after
out=pathlib.Path('artifacts/action-grid-regression-20260929');out.mkdir(exist_ok=True)
(out/'audit-before.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

"""Neutral branch: old/new CPU and DirectML update comparison on real scenes."""
import sys,pathlib,json,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3.common.logger import configure
from touhou_ai.live_action_grid import build_live_action_model,LiveActionGridContract
from touhou_ai.dual_grid import DualGridContract,GridRolloutBuffer
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.checkpoint_rng import capture_rng,restore_rng
from touhou_ai.directml_update import update
torch.set_num_threads(1)
out=pathlib.Path('artifacts/action-grid-regression-20260929/update-parity');out.mkdir(exist_ok=True)
profile=json.loads(pathlib.Path('configs/sharu-inspired-v1.json').read_text())
settings=profile['provisional_ppo']|profile['evasion_ppo_overrides']|dict(n_steps=32,batch_size=16,n_epochs=1,target_kl=None)
old=SeparateClipPPO(EvasionPolicy,DualGridContract(),device='cpu',rollout_buffer_class=GridRolloutBuffer,**settings,
    policy_kwargs=dict(share_features_extractor=False,features_extractor_class=NarrowGridFeatures,net_arch=dict(pi=[256,128],vf=[256,128])))
new=build_live_action_model(settings)
with torch.no_grad():
    for branch in (new.policy.pi_features_extractor.action,new.policy.vf_features_extractor.action):
        for p in branch.parameters():p.zero_()
raws=json.loads(pathlib.Path('artifacts/live-action-diagnosis-20260929/raws.json').read_text())[:32]
c=LiveActionGridContract();obs=[c.encode(r) for r in raws]
for m in (old,new):
    m.set_logger(configure(None,[]))
    for i,o in enumerate(obs):
        x={k:v for k,v in o.items() if k in m.observation_space.spaces}
        action=np.array([[i%9,0,i%2,0]])
        with torch.no_grad():v,lp,_=m.policy.evaluate_actions(m.policy.obs_to_tensor(x)[0],torch.as_tensor(action))
        m.rollout_buffer.add(x,action,np.array([-60 if i in (15,31) else 0]),np.array([i==0]),v,lp)
    m.rollout_buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]))
    m._current_progress_remaining=1.
state=capture_rng();models={};infos={}
for name,m in (('old_cpu',copy.deepcopy(old)),('new_cpu',copy.deepcopy(new)),('old_gpu',copy.deepcopy(old)),('new_gpu',copy.deepcopy(new))):
    restore_rng(state)
    if name.endswith('gpu'):
        m,info=update(m,m.rollout_buffer,out/name);assert info['backend']=='directml',info;infos[name]=info
    else:m.train()
    models[name]=m
    infos[name]=infos.get(name,{})|{'rng_after':capture_rng()}
def error(a,b):
    errors=[]
    for k,v in a.policy.state_dict().items():
        w=b.policy.state_dict()[k]
        if v.shape!=w.shape:w=w[:,:v.shape[1]]
        errors.append(float((v-w).abs().max()))
    return max(errors)
result=dict(neutral_cpu_update_error=error(models['old_cpu'],models['new_cpu']),
    neutral_gpu_update_error=error(models['old_gpu'],models['new_gpu']),
    old_cpu_gpu_error=error(models['old_cpu'],models['old_gpu']),new_cpu_gpu_error=error(models['new_cpu'],models['new_gpu']),
    rng_continuity=all(x['rng_after']==infos['old_cpu']['rng_after'] for x in infos.values()))
assert result['neutral_cpu_update_error']<1e-5 and result['neutral_gpu_update_error']<1e-5,result
assert result['old_cpu_gpu_error']<1e-4 and result['new_cpu_gpu_error']<1e-4,result
assert result['rng_continuity'],result
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

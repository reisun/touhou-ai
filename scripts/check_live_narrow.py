"""Preflight the exact live evasion architecture and isolated DirectML update."""
import json,pathlib,numpy as np,torch
from stable_baselines3.common.logger import configure
from touhou_ai.dual_grid import DualGridContract,GridRolloutBuffer
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.live_model import model_config,algorithm_class
from touhou_ai.directml_update import update
out=pathlib.Path('artifacts/live-narrow-preflight-20260929');out.mkdir(exist_ok=True)
torch.set_num_threads(1);profile=json.loads(pathlib.Path('configs/sharu-inspired-v1.json').read_text());profile['evasion_policy_overrides']={'share_features_extractor':False,'cnn_architecture':'narrow_grid','algorithm':'SeparateClipPPO'}
selection=model_config(profile,True,True)
try:model_config(profile,True,True,{'contract':'th10-dual-grid-v6'})
except ValueError:pass
else:raise AssertionError('old resume accepted')
assert model_config(profile,True,True,{'contract':'th10-dual-grid-v6',**selection})==selection
c=DualGridContract();settings=profile['provisional_ppo']|profile['evasion_ppo_overrides']|{'n_steps':32,'batch_size':16}
m=algorithm_class(selection['algorithm'])(EvasionPolicy,c,device='cpu',rollout_buffer_class=GridRolloutBuffer,policy_kwargs={'share_features_extractor':False,'features_extractor_class':NarrowGridFeatures,'net_arch':{'pi':[256,128],'vf':[256,128]}},**settings)
m.set_logger(configure(str(out),['json']));rng=np.random.default_rng(83)
for i in range(32):
 o={k:np.zeros(s.shape,dtype=np.float32) for k,s in c.observation_space.spaces.items()};o['player'][:]=rng.random(o['player'].shape);o['bomb_clock'][0]=(i%6)/6
 with torch.no_grad():
  t,_=m.policy.obs_to_tensor(o);a,v,lp,ds=m.policy.forward_with_distribution(t);_,lp2,en=m.policy.evaluate_actions(t,a)
 assert a[0,1]==a[0,3]==0 and ds[1].probs[0,1]==ds[3].probs[0,1]==0
 torch.testing.assert_close(lp,lp2);assert torch.isfinite(en).all()
 m.rollout_buffer.add(o,a.numpy(),np.array([-60. if i==31 else 0.]),np.array([i==0]),v,lp)
m.rollout_buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]));m._current_progress_remaining=1.
assert m.policy.pi_features_extractor is not m.policy.vf_features_extractor
m,info=update(m,m.rollout_buffer,out/'directml')
assert info['backend']=='directml',info
assert type(m).__name__=='SeparateClipPPO' and isinstance(m.policy.pi_features_extractor,NarrowGridFeatures)
metrics=m.logger.name_to_value
assert 'train/actor_gradient_norm' in metrics and 'train/critic_gradient_norm' in metrics
m.save(out/'verified');restored=algorithm_class(selection['algorithm']).load(out/'verified',device='cpu')
assert type(restored) is type(m) and isinstance(restored.policy.vf_features_extractor,NarrowGridFeatures)
result={'selection':selection,'directml':info,'parameters':sum(p.numel() for p in m.policy.parameters()),'actor_gradient_norm':float(metrics['train/actor_gradient_norm']),'critic_gradient_norm':float(metrics['train/critic_gradient_norm']),'forced_shot_bomb_zero':True,'reload_verified':True}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

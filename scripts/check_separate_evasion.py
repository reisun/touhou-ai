import tempfile,pathlib,numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.dual_grid import DualGridContract,DualGridFeatures,GridRolloutBuffer
from touhou_ai.evasion_only import EvasionPolicy,DeathOnlyRewards
from stable_baselines3.common.logger import configure
torch.set_num_threads(1);c=DualGridContract();m=PPO(EvasionPolicy,c,n_steps=32,batch_size=16,n_epochs=1,rollout_buffer_class=GridRolloutBuffer,policy_kwargs={'share_features_extractor':False,'features_extractor_class':DualGridFeatures,'net_arch':{'pi':[256,128],'vf':[256,128]}},device='cpu');m.set_logger(configure(None,[]));o={k:np.zeros(s.shape,dtype=np.float32) for k,s in c.observation_space.spaces.items()}
for i in range(32):
 o['bomb_clock'][0]=(i%6)/6
 with torch.no_grad():
  t,_=m.policy.obs_to_tensor(o);a,v,lp,ds=m.policy.forward_with_distribution(t);v2,lp2,en=m.policy.evaluate_actions(t,a)
 assert a[0,1]==0 and a[0,3]==0 and ds[1].probs[0,1]==0 and ds[3].probs[0,1]==0
 assert torch.allclose(lp,lp2) and torch.isfinite(en).all()
 m.rollout_buffer.add(o,a.numpy(),np.array([-60. if i==31 else 0.]),np.array([i==0]),v,lp)
m.rollout_buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]));m._current_progress_remaining=1.;m.train()
assert all(torch.isfinite(p).all() for p in m.policy.parameters())
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d)/'model';m.save(p);n=PPO.load(p,device='cpu');a,_=n.predict(o);assert a[1]==a[3]==0
r=DeathOnlyRewards();r.reset('x');v,com=r.calculate('x',[{'id':'hit1','confirmed':True,'kind':'hit'},{'id':'damage1','kind':'damage','amount':999999},{'kind':'progress'}]);assert v==-60 and com==dict(damage=0.,progress=0.,hit=-60.,power_down=0.)
assert r.calculate('x',[{'id':'hit1','confirmed':True,'kind':'hit'}])[0]==0
print('PASS: forced actions, rollout/evaluation log-prob consistency, finite entropy, PPO update, save/reload, death-only reward and deduplication')

assert m.policy.pi_features_extractor is not m.policy.vf_features_extractor
m.policy.optimizer.zero_grad()
t,_=m.policy.obs_to_tensor(o)
values,lp,ent=m.policy.evaluate_actions(t,torch.tensor([[0,0,1,0]]))
values.sum().backward()
assert all(p.grad is None or not p.grad.any() for p in m.policy.pi_features_extractor.parameters())
assert any(p.grad is not None and p.grad.any() for p in m.policy.vf_features_extractor.parameters())
m.policy.optimizer.zero_grad()
values,lp,ent=m.policy.evaluate_actions(t,torch.tensor([[0,0,1,0]]))
(-lp.sum()).backward()
assert all(p.grad is None or not p.grad.any() for p in m.policy.vf_features_extractor.parameters())
assert any(p.grad is not None and p.grad.any() for p in m.policy.pi_features_extractor.parameters())
assert not n.policy.share_features_extractor
print('PASS gradient isolation both directions and persisted separate extractors')

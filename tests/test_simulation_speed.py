import unittest,numpy as np,torch,tempfile,pathlib
from stable_baselines3 import PPO
from tests.fixed_dodge_reference import FixedDodge as Reference
from touhou_ai.fixed_dodge import FixedDodge
from touhou_ai.side_dodge import SideGrid
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.simulation_speed import distributions,assess_batch,tune_cpu
class SpeedTests(unittest.TestCase):
 def test_exact_physics_and_observations(self):
  a,b=Reference(),FixedDodge();rng=np.random.default_rng(9)
  def reset(side):
   for e in [a,b]:
    e.reset()
    if side<0:
     for bullet in e.bullets:bullet['position'][0]*=-1
     e.bullets.sort(key=lambda x:x['position'][0])
  reset(1)
  for i in range(1000):
   action=[int(rng.integers(9)),0,int(rng.integers(2)),0]
   oa,ra,da,ta,ia=a.step(action);ob,rb,db,tb,ib=b.step(action)
   self.assertEqual((ra,da,ta,ia),(rb,db,tb,ib));np.testing.assert_array_equal(a.pos,b.pos);self.assertEqual(a.bullets,b.bullets)
   for k in oa:np.testing.assert_array_equal(oa[k],ob[k])
   if da:reset(-1 if i%2 else 1)
 def test_inference_batch_and_save(self):
  torch.set_num_threads(1);m=PPO(EvasionPolicy,SideGrid(),seed=7,n_steps=8,batch_size=8,n_epochs=1,policy_kwargs={'features_extractor_class':NarrowGridFeatures,'share_features_extractor':False,'net_arch':{'pi':[64,64],'vf':[64,64]}})
  o,_=SideGrid().reset();t,_=m.policy.obs_to_tensor(o)
  with torch.no_grad():
   ref=m.policy.get_distribution(t).distribution;fast=distributions(m.policy,o)
   for x,y in zip(ref,fast):torch.testing.assert_close(x.probs,y.probs,rtol=0,atol=0)
  self.assertEqual(assess_batch(m,SideGrid,n=4,side=-1,batch_size=1),assess_batch(m,SideGrid,n=4,side=-1,batch_size=4))
  tune_cpu(m);m.learn(8);self.assertEqual(torch.get_num_threads(),1)
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'model';m.save(p);n=PPO.load(p,device='cpu');self.assertNotIn('train',n.__dict__);self.assertNotIn('_excluded_save_params',n.__dict__)
   np.testing.assert_array_equal(m.predict(o,deterministic=True)[0],n.predict(o,deterministic=True)[0])

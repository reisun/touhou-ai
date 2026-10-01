import unittest
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import NumericalAblation,SpeedPolicy
class Tests(unittest.TestCase):
 def test_relative_preserves_physics_and_padding(self):
  a=NumericalAblation();r=NumericalAblation(relative=True)
  a.reset(seed=55);r.reset(seed=55)
  for _ in range(80):
   oa,ra,da,_,_=a.step([6,0,1,0]);orr,rr,dr,_,_=r.step([6,0,1,0])
   np.testing.assert_array_equal(a.xy,r.xy);self.assertEqual((ra,da),(rr,dr))
   x=oa['state'][5:].reshape(-1,5);y=orr['state'][5:].reshape(-1,5);mask=x[:,4]>0
   np.testing.assert_allclose(y[mask,:2],x[mask,:2]-a.pos/[192,448],atol=2e-7)
   np.testing.assert_array_equal(y[~mask],0);np.testing.assert_array_equal(x[:,2:],y[:,2:])
   self.assertTrue(r.observation_space.contains(orr))
   if da:break
 def test_fixed_focus_distribution_and_step(self):
  torch.set_num_threads(1)
  for focus in [0,1]:
   e=NumericalAblation(fixed_focus=focus);o,_=e.reset(seed=7)
   self.assertEqual(e.focus,focus)
   m=PPO(SpeedPolicy,e,n_steps=8,batch_size=8,policy_kwargs={'fixed_focus':focus,'net_arch':[8]},device='cpu')
   tensor,_=m.policy.obs_to_tensor(o);d=m.policy.get_distribution(tensor)
   self.assertEqual(float(d.distribution[2].probs[0,focus]),1.)
   self.assertEqual(float(d.distribution[2].entropy()[0]),0.)
   a,_=m.predict(o);self.assertEqual(a[2],focus);e.step(a)
   with self.assertRaises(ValueError):e.step([0,0,1-focus,0])
if __name__=='__main__':unittest.main()

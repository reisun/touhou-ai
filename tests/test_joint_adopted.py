import unittest,tempfile,pathlib
import numpy as np,torch
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.joint_movement import JointMovementPolicy
from touhou_ai.autumn_ablation import GridAblation
from touhou_ai.narrow_grid import NarrowGridFeatures
from scripts.compare_joint_adopted import decode_joint,assess
class Tests(unittest.TestCase):
 def test_factorized_sampling_matches_independent(self):
  rng=np.random.default_rng(42)
  for _ in range(100):
   p=rng.dirichlet(np.ones(9));q=rng.dirichlet(np.ones(2));u=rng.random(4)
   self.assertEqual(decode_joint((p[:,None]*q).ravel(),u),[int(np.searchsorted(p.cumsum(),u[0])),0,int(np.searchsorted(q.cumsum(),u[2])),0])
 def test_joint_cnn_separate_clipping_and_resume(self):
  torch.set_num_threads(1)
  m=SeparateClipPPO(JointMovementPolicy,GridAblation(),seed=7,n_steps=8,batch_size=8,n_epochs=1,policy_kwargs={'share_features_extractor':False,'features_extractor_class':NarrowGridFeatures,'net_arch':{'pi':[256,128],'vf':[256,128]}},device='cpu')
  m.learn(8);m.clip_separate_gradients()
  with tempfile.TemporaryDirectory() as td:
   path=pathlib.Path(td)/'m';m.save(path);r=SeparateClipPPO.load(path,env=GridAblation(),device='cpu');r.learn(8,reset_num_timesteps=False);self.assertEqual(r.num_timesteps,16)
   for mode in ['sample','greedy','frozen_sample','frozen_greedy']:
    e=assess(r,GridAblation,n=1,mode=mode);self.assertEqual(e['n'],1);self.assertEqual(e['episodes'][0]['seed'],5000)
if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from touhou_ai.avoidance_augmentation import reflect_batch
from touhou_ai.autumn_two_sets import TwoSetsGrid
class Tests(unittest.TestCase):
 def test_matches_reencoded_mirrored_scene(self):
  e=TwoSetsGrid();e.reset(seed=11);e.pos[0]=37.;o=e.observe();batch={k:np.stack([v]) for k,v in o.items()}
  reflect_batch(batch,np.array([[3,0,1,0]]),np.array([0]))
  e.pos[0]*=-1;e.xy[:,0]*=-1;e.vel[:,0]*=-1;mirrored=e.observe()
  for k in batch:np.testing.assert_allclose(batch[k][0],mirrored[k],atol=1e-6,err_msg=k)
 def test_reflection_twice_restores_everything(self):
  o,_=TwoSetsGrid().reset(seed=1);batch={k:np.stack([v,v]) for k,v in o.items()};original={k:v.copy() for k,v in batch.items()};a=np.array([[3,0,1,0],[8,0,0,0]]);orig=a.copy()
  reflect_batch(batch,a,np.array([0]));self.assertEqual(a[0,0],7);self.assertEqual(a[1,0],8)
  reflect_batch(batch,a,np.array([0]))
  np.testing.assert_array_equal(a,orig)
  for k in batch:np.testing.assert_array_equal(batch[k],original[k])
if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from touhou_ai.relative_autumn import RelativeTwoSetsNumerical
class Tests(unittest.TestCase):
 def test_all_bullets_retained(self):
  e=RelativeTwoSetsNumerical();o,_=e.reset(seed=3)
  self.assertTrue(e.observation_space.contains(o))
  self.assertEqual(o['state'][5:].reshape(-1,5)[:,4].sum(),152)
 def test_relative_translation(self):
  e=RelativeTwoSetsNumerical();a,_=e.reset(seed=3)
  e.pos+=np.array([8,0]);e.xy[e.alive]+=np.array([8,0]);b=e.observe()
  np.testing.assert_allclose(a['state'][5:],b['state'][5:],atol=1e-6)
if __name__=='__main__':unittest.main()

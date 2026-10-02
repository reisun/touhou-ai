import unittest
import numpy as np
from touhou_ai.dual_grid import paint_laser,laser_patch
from tests.test_laser_paint_batch import reference
class LaserCache(unittest.TestCase):
 def test_exact_keys_and_isolated_output(self):
  laser_patch.cache_clear()
  c=dict(active=True,origin=[-10.,30.],angle=.4,length=120.,width=8.,field_validated=True)
  for change,origin in [({},[-192.,0.]),({},[-192.,0.]),({'width':9.},[-192.,0.]),({'angle':.400000001},[-192.,0.]),({'origin':[-9.9,30.]},[-192.,0.]),({},[-191.9,0.]),({'field_validated':False},[-192.,0.]),({'active':False},[-192.,0.])]:
   a=np.zeros((2,56,48),np.float32);b=a.copy();x=c|change;o=np.array(origin)
   paint_laser(a,0,1,x,o,8);reference(b,0,1,x,o,8);np.testing.assert_array_equal(a,b)
   a.fill(7) # output mutation must not corrupt cached coverage
  self.assertGreaterEqual(laser_patch.cache_info().hits,2)
  self.assertEqual(laser_patch.cache_info().maxsize,256)

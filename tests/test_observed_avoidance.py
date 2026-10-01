import unittest
import numpy as np
from touhou_ai.observed_avoidance import choose
class Tests(unittest.TestCase):
 def test_no_threat_holds(self):
  a=choose(np.array([0.,330.]),np.empty((0,2)),np.empty((0,2)))
  self.assertEqual(a[0],0);self.assertEqual(a[1],0);self.assertEqual(a[3],0)
 def test_imminent_bullet_moves(self):
  a=choose(np.array([0.,330.]),np.array([[0.,318.]]),np.array([[0.,3.]]),4)
  self.assertNotEqual(a[0],0)
 def test_order_invariance(self):
  xy=np.array([[0.,318.],[15.,325.],[-15.,320.]])
  v=np.array([[0.,3.],[0.,2.],[0.,1.]])
  np.testing.assert_array_equal(choose(np.array([0.,330.]),xy,v,4),choose(np.array([0.,330.]),xy[::-1],v[::-1],4))
if __name__=='__main__':unittest.main()

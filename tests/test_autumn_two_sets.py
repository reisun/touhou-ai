import unittest
import numpy as np
from touhou_ai.autumn_two_sets import TwoSetsNumerical
class Tests(unittest.TestCase):
 def test_all_four_waves_clear_before_success(self):
  e=TwoSetsNumerical();e.reset(seed=1000);original=e._frame_velocity
  def ghost():
   ids,mask=original();return ids,np.zeros_like(mask)
  e._frame_velocity=ghost
  while True:
   _,_,done,trunc,info=e.step([0,0,0,0])
   if e.frame==600:self.assertFalse(done)
   if done or trunc:break
  self.assertTrue(info['success']);self.assertFalse(trunc);self.assertGreater(e.frame,600)
  self.assertEqual(len(e.emissions),4);self.assertEqual(info['sets_cleared'],2);self.assertFalse(e.alive.any())
 def test_timeout_is_not_success(self):
  e=TwoSetsNumerical();e.horizon=2;e.reset(seed=1000)
  _,_,done,trunc,info=e.step([0,0,0,0])
  self.assertFalse(done);self.assertTrue(trunc);self.assertFalse(info['success'])
 def test_death_not_completion(self):
  e=TwoSetsNumerical();e.reset(seed=1000)
  while True:
   _,_,done,trunc,info=e.step([0,0,0,0])
   if done or trunc:break
  self.assertTrue(e.dead);self.assertFalse(info['success'])
if __name__=='__main__':unittest.main()

import unittest,numpy as np
from touhou_ai.avoidance_rewards_sim import RewardGrid,RewardNumerical
class RewardTests(unittest.TestCase):
 def test_death_and_success(self):
  for cls in [RewardGrid,RewardNumerical]:
   for side in [-1,1]:
    e=cls();e.reset(options={'side':side})
    for i in range(80):
     _,_,d,_,info=e.step([3 if side==1 and 12<=i<17 else 7 if 12<=i<17 else 0,0,0,0])
     if d:break
    self.assertTrue(info['success']);self.assertEqual(e.reward_totals,{'hit':0.,'risk':2.,'pass':4.})
    e.reset(options={'side':side})
    for _ in range(80):
     _,_,d,_,info=e.step([0,0,0,0])
     if d:break
    self.assertFalse(info['success']);self.assertEqual(e.reward_totals,{'hit':-60.,'risk':0.,'pass':0.})
 def test_caps(self):
  rng=np.random.default_rng(7);e=RewardNumerical()
  for _ in range(50):
   e.reset()
   for i in range(80):
    _,_,d,_,info=e.step([int(rng.integers(9)),0,int(rng.integers(2)),0])
    self.assertLessEqual(e.reward_totals['risk'],2);self.assertLessEqual(e.reward_totals['pass'],4)
    if d:break

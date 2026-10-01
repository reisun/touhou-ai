import unittest
import numpy as np
from touhou_ai.side_dodge import SideGrid,SideNumerical
class SideDodgeTests(unittest.TestCase):
 def test_reachable_and_opposite_observation(self):
  for cls in [SideGrid,SideNumerical]:
   for side in [-1,1]:
    e=cls();o,_=e.reset(options={'side':side});self.assertTrue(e.observation_space.contains(o))
    before=[b['position'][:] for b in e.bullets];opp=e.opposite_observation();self.assertEqual(before,[b['position'] for b in e.bullets])
    other=cls();normal,_=other.reset(options={'side':-side})
    for k in opp:np.testing.assert_array_equal(opp[k],normal[k])
    for i in range(80):
     _,_,d,_,info=e.step([3 if side==1 and i<5 else 7 if i<5 else 0,0,0,0])
     if d:break
    self.assertTrue(info['success'])
 def test_seeded_side_sequence(self):
  a,b=SideGrid(),SideGrid();a.reset(seed=11);b.reset(seed=11);s=[]
  for _ in range(40):
   a.reset();b.reset();self.assertEqual(a.side,b.side);s.append(a.side)
  self.assertEqual(set(s),{-1,1})

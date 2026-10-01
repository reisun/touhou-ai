import unittest,numpy as np
from touhou_ai.corridor_dodge import CorridorGrid,CorridorNumerical
class CorridorTests(unittest.TestCase):
 def play(self,cls,side,mode):
  e=cls();o,_=e.reset(options={'side':side});self.assertTrue(e.observation_space.contains(o));positions=[]
  for i in range(80):
   move=0
   if mode=='large':move=3
   if mode=='fine':
    if i<2:move=3 if side==1 else 7
    if 21<=i<25:move=7 if side==1 else 3
   o,r,d,_,info=e.step([move,0,1 if mode=='fine' else 0,0]);positions.append(e.pos.copy())
   if d:return info['success'],positions
 def test_requires_fine_steering(self):
  for cls in [CorridorGrid,CorridorNumerical]:
   for side in [-1,1]:
    self.assertFalse(self.play(cls,side,'stop')[0]);self.assertFalse(self.play(cls,side,'large')[0]);ok,ps=self.play(cls,side,'fine');self.assertTrue(ok);self.assertLessEqual(max(abs(p[0]) for p in ps),8)
 def test_representations_same_physics(self):
  a,b=CorridorGrid(),CorridorNumerical();a.reset(seed=3);b.reset(seed=3);rng=np.random.default_rng(7)
  for i in range(500):
   action=[int(rng.integers(9)),0,int(rng.integers(2)),0];oa,ra,da,_,ia=a.step(action);ob,rb,db,_,ib=b.step(action)
   self.assertEqual((ra,da,ia),(rb,db,ib));np.testing.assert_array_equal(a.pos,b.pos);self.assertEqual(a.bullets,b.bullets)
   self.assertTrue(a.observation_space.contains(oa));self.assertTrue(b.observation_space.contains(ob))
   if da:a.reset();b.reset()
 def test_mirrored_observations(self):
  for cls in [CorridorGrid,CorridorNumerical]:
   a,b=cls(),cls();a.reset(options={'side':-1});o,_=b.reset(options={'side':1});mirrored=a.opposite_observation()
   for k in o:np.testing.assert_array_equal(o[k],mirrored[k])

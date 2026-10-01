import unittest
import numpy as np
from touhou_ai.autumn_sky import AutumnSky,AutumnNumerical,TURNS,MAX_BULLETS
class Tests(unittest.TestCase):
 def test_emitter_and_turns(self):
  e=AutumnNumerical();e.reset(seed=99)
  self.assertEqual(e.n,152)
  np.testing.assert_allclose(np.diff(e.angle[:38]),2*np.pi/38)
  angle=e.angle[0]
  for age,speed,turn in [(0,1,0),(45,1,0),(74,2/60,0),(75,2,TURNS[0,0]),(136,2,0),(197,1.2,TURNS[0,2])]:
   e.frame=age;e._frame_velocity()
   np.testing.assert_allclose(e.vel[0],np.array([np.cos(angle+turn),np.sin(angle+turn)])*speed,atol=1e-8)
 def test_observation_and_dynamics_parity(self):
  g=AutumnSky();n=AutumnNumerical();rng=np.random.default_rng(5)
  for seed in range(3):
   og,_=g.reset(seed=seed);on,_=n.reset(seed=seed)
   for t in range(300):
    a=[int(rng.integers(9)),0,int(rng.integers(2)),0]
    og,rg,dg,_,ig=g.step(a);on,rn,dn,_,inn=n.step(a)
    self.assertTrue(g.observation_space.contains(og));self.assertTrue(n.observation_space.contains(on))
    np.testing.assert_array_equal(g.xy,n.xy);np.testing.assert_array_equal(g.pos,n.pos)
    self.assertEqual((rg,dg,ig),(rn,dn,inn))
    if dg:break
 def test_offscreen_top_retention(self):
  e=AutumnNumerical();e.reset(seed=7);e.frame=100
  e.xy[0]=[0,-50];e.xy[1]=[0,-71];e.xy[2]=[200,100];e._cull()
  self.assertTrue(e.alive[0]);self.assertFalse(e.alive[1]);self.assertFalse(e.alive[2])
 def test_full_horizon_and_capacity(self):
  e=AutumnNumerical();e.reset(seed=7)
  # Suppress collisions only for emission-schedule accounting.
  for frame in range(600):
   e.frame=frame
   if frame and frame%120==0:e._emit()
   ids,_=e._frame_velocity();e.xy[ids]+=e.vel[ids]
  self.assertEqual(e.n,MAX_BULLETS)
  self.assertEqual([x['frame'] for x in e.emissions],[0,120,240,360,480])
  self.assertEqual(e.group[152],4)
if __name__=='__main__':unittest.main()

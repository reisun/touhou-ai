import unittest
import numpy as np
from touhou_ai.fixed_dodge import FixedDodge

class FixedDodgeTests(unittest.TestCase):
    def run_path(self,right):
        e=FixedDodge();obs,_=e.reset();self.assertTrue(e.observation_space.contains(obs));reward=0
        for i in range(80):
            obs,r,done,_,info=e.step([3 if right and i<5 else 0,0,0,0]);reward+=r
            if done:return reward,info['success']
        self.fail('task did not terminate')
    def test_stationary_dies(self):self.assertEqual(self.run_path(False),(-60,False))
    def test_gap_is_reachable(self):self.assertEqual(self.run_path(True),(0,True))
    def test_reset_reproducible(self):
        e=FixedDodge();a,_=e.reset();e.step([3,0,1,0]);b,_=e.reset()
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_disabled_actions(self):
        e=FixedDodge();e.reset()
        with self.assertRaises(ValueError):e.step([0,1,0,0])

class NumericalDodgeTests(unittest.TestCase):
    def test_matches_grid_physics(self):
        from touhou_ai.numerical_dodge import NumericalDodge
        x=FixedDodge();y=NumericalDodge();x.reset();o,_=y.reset()
        self.assertTrue(y.observation_space.contains(o));rng=np.random.default_rng(1)
        for _ in range(300):
            a=[int(rng.integers(9)),0,int(rng.integers(2)),0]
            _,rx,dx,_,ix=x.step(a);oy,ry,dy,_,iy=y.step(a)
            self.assertEqual((rx,dx,ix),(ry,dy,iy));np.testing.assert_array_equal(x.pos,y.pos)
            self.assertEqual(x.bullets,y.bullets);self.assertTrue(y.observation_space.contains(oy))
            if dx:x.reset();y.reset()

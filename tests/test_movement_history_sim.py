import unittest
import numpy as np
from touhou_ai.movement_history_sim import HistoryEnv
from touhou_ai.fixed_dodge import DIRECTIONS

class MovementHistoryTests(unittest.TestCase):
    def test_optimized_observation_matches_original_simulator(self):
        from scripts.live_jitter_trial import Env
        env=HistoryEnv();reference=Env(.1)
        rng=np.random.default_rng(57)
        for seed in range(3):
            env.reset(seed=seed);reference.reset(seed=seed)
            for i in range(300):
                action=[int(rng.integers(9)),0,int(rng.integers(2)),0]
                a,r,d,t,info=env.step(action);b,s,e,u,other=reference.step(action)
                self.assertEqual((r,d,t,info),(s,e,u,other))
                for key in b:np.testing.assert_array_equal(a[key],b[key])
                if d or t:break

    def test_selected_history_order_reset_and_zero_control(self):
        env=HistoryEnv();control=HistoryEnv(history=False)
        a,_=env.reset(seed=7);b,_=control.reset(seed=7)
        self.assertEqual(a['movement_history'].shape,(6,4))
        self.assertFalse(a['movement_history'].any())
        actions=[[i%9,0,i%2,0] for i in range(8)]
        for action in actions:
            a,r,d,t,_=env.step(action);b,s,e,u,_=control.step(action)
            self.assertEqual((r,d,t),(s,e,u))
            for key in a:
                if key!='movement_history':np.testing.assert_array_equal(a[key],b[key])
            self.assertFalse(b['movement_history'].any())
        expected=np.array([[*DIRECTIONS[x[0]],x[2],1] for x in actions[-6:]],dtype=np.float32)
        np.testing.assert_array_equal(a['movement_history'],expected)
        a,_=env.reset(seed=7);self.assertFalse(a['movement_history'].any())

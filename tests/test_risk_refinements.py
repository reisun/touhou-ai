import unittest
import numpy as np
import torch
from touhou_ai.risk_refinements import forecast, RefinedEnv, build
from touhou_ai.spatial_input_candidates import risk_grid

class Refinements(unittest.TestCase):
    def test_binary_parity_and_timing(self):
        e=RefinedEnv();e.reset(seed=7)
        rng=np.random.default_rng(23)
        for _ in range(120):
            raw=e.raw_observation()
            np.testing.assert_array_equal(forecast(raw),risk_grid(raw))
            graded=forecast(raw,graded=True)
            np.testing.assert_array_equal(graded>0,risk_grid(raw)>0)
            np.testing.assert_array_equal(forecast(raw,continuous=True)[:2],risk_grid(raw)[:2])
            for continuous in (False,True):
                np.testing.assert_array_equal(forecast(raw,continuous=continuous,graded=True),forecast(raw,continuous=continuous,graded=True,prune=False))
            _,_,done,_,_=e.step([rng.integers(9),0,rng.integers(2),0])
            if done:e.reset(seed=8)
        raw['player']['position']=[0,300]
        raw['bullets']=[dict(position=[10,300],velocity_raw=[-1,0],hitbox_raw=[4,4],flags_raw=2)]
        self.assertAlmostEqual(float(forecast(raw,graded=True)[2,1,1]),1-7/17,places=6)
        self.assertEqual(forecast(raw,continuous=True)[2,1,0],0)
        self.assertEqual(forecast(raw)[2,1,0],1)

    def test_reward_only_and_initial_weights(self):
        a=RefinedEnv();b=RefinedEnv('scale')
        a.reset(seed=7);b.reset(seed=7)
        for _ in range(300):
            x,r,d,t,i=a.step([0,0,0,0]);y,s,*_=b.step([0,0,0,0])
            self.assertEqual(r/60,s)
            for k in x:np.testing.assert_array_equal(x[k],y[k])
            if d:break
        torch.set_num_threads(1)
        m,_,_=build('control',7)
        for variant in ('urgency','continuous','scale'):
            n,_,_=build(variant,7)
            for k,v in m.policy.state_dict().items():torch.testing.assert_close(v,n.policy.state_dict()[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

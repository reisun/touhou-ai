import unittest
import numpy as np,torch
from touhou_ai.autumn_training import build_model,build_spatial_reference_model
from touhou_ai.risk_refinements import build

class AdoptedReward(unittest.TestCase):
    def test_matches_verified_scale_and_frozen_reference(self):
        torch.set_num_threads(1)
        for seed in (7,17,27):
            m,cls,config=build_model('cnn',seed);old,other,_=build('scale',seed)
            for k,v in m.policy.state_dict().items():torch.testing.assert_close(v,old.policy.state_dict()[k],rtol=0,atol=0)
            a=cls();b=other();a.reset(seed=7);b.reset(seed=7)
            for _ in range(300):
                x,r,d,t,i=a.step([0,0,0,0]);y,s,*_=b.step([0,0,0,0])
                self.assertEqual(r,s)
                for k in x:np.testing.assert_array_equal(x[k],y[k])
                if d:break
            self.assertEqual(r,-1)
        _,ref,c=build_spatial_reference_model('cnn',7)
        self.assertEqual(c['version'],'autumn-baseline-v3')
        e=ref();e.reset(seed=7)
        for _ in range(300):
            _,r,d,_,_=e.step([0,0,0,0])
            if d:break
        self.assertEqual(r,-60)

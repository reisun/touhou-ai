"""Random left/right gaps; no side label in the policy observation."""
import numpy as np
from touhou_ai.fixed_dodge import FixedDodge
from touhou_ai.numerical_dodge import NumericalDodge
class SideGap:
    def reset(self,*,seed=None,options=None):
        if seed is not None or not hasattr(self,'side_rng'):self.side_rng=np.random.default_rng(7 if seed is None else seed)
        super().reset(seed=seed)
        self.side=(options or {}).get('side',None)
        if self.side is None:self.side=int(self.side_rng.choice([-1,1]))
        if self.side not in [-1,1]:raise ValueError('side must be -1 or1')
        if self.side==-1:
            for b in self.bullets:b['position'][0]*=-1
            self.bullets.sort(key=lambda b:b['position'][0])
        return self.observe(),{}
    def opposite_observation(self):
        saved=self.bullets
        self.bullets=sorted([{**b,'position':[-b['position'][0],b['position'][1]]} for b in saved],key=lambda b:b['position'][0])
        try:return self.observe()
        finally:self.bullets=saved
class SideGrid(SideGap,FixedDodge):pass
class SideNumerical(SideGap,NumericalDodge):
    def observe(self):
        o=super().observe();o['state'][2:4]=0
        return o

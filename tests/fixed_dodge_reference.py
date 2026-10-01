"""Deterministic synthetic wall-gap task; no game process or input access."""
import numpy as np
from touhou_ai.dual_grid import DualGridContract

DIRECTIONS=np.array([[0,0],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]],dtype=float)
DIRECTIONS[[2,4,6,8]]/=np.sqrt(2)

class FixedDodge(DualGridContract):
    def reset(self, *, seed=None, options=None):
        # Exactly the same scene every episode, including evaluation.
        self.pos=np.array([0.,330.]);self.frame=0;self.focus=0;self.dead=False
        self.bullets=[{'position':[float(x),210.], 'velocity_raw':[0.,3.],
                      'hitbox_raw':[8.,8.], 'flags_raw':2} for x in range(-192,193,8) if not 32<=x<=64]
        return self.observe(),{}

    def observe(self):
        raw={'stage':1,'stage_frame':self.frame,'lives_raw':0,'power_raw':0,
             'player':{'position':self.pos.tolist(),'velocity_raw':[0,0],'hitbox_raw':[1,1],
                       'status':4 if self.dead else 1,'invincibility_raw':0,'focus_raw':self.focus},
             'bullets':self.bullets,'enemies':[],'items':[],'lasers':[],'player_shots':[],
             'bomb':{'state':0},'spell':None}
        return self.encode(raw)

    def step(self,action):
        if action[1] or action[3]:raise ValueError('shot/bomb must be disabled')
        self.focus=int(action[2]);velocity=DIRECTIONS[int(action[0])]*(2 if self.focus else 4.5)
        # Sub-frame checks avoid tunnelling; full bullet size vs player half extent.
        for _ in range(8):
            self.pos=np.clip(self.pos+velocity*.25,[-184,32],[184,432])
            for b in self.bullets:
                b['position'][1]+=.75
                if np.all(np.abs(self.pos-np.array(b['position']))<=5):self.dead=True
            if self.dead:break
        self.frame+=2
        done=self.dead or self.frame>=160
        return self.observe(), -60. if self.dead else 0.,done,False,{'success':done and not self.dead}

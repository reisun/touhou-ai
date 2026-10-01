"""Direct numerical observation baseline for the identical FixedDodge physics."""
import gymnasium as gym
import numpy as np
from touhou_ai.fixed_dodge import FixedDodge,DIRECTIONS

class NumericalDodge(FixedDodge):
    def __init__(self):
        super().__init__()
        self.observation_space=gym.spaces.Dict({'state':gym.spaces.Box(-2,2,(181,),dtype=np.float32),'bomb_clock':gym.spaces.Box(0,1,(1,),dtype=np.float32)})
    def reset(self,**kwargs):
        self.velocity=np.zeros(2)
        return super().reset(**kwargs)
    def step(self,action):
        self.velocity=DIRECTIONS[int(action[0])]*(2 if action[2] else 4.5)
        return super().step(action)
    def observe(self):
        # 5 player values + 44 bullets * (x,y,vx,vy). No gap label or oracle action.
        values=[self.pos[0]/192,self.pos[1]/448,*(self.velocity/4.5),self.focus]
        for b in self.bullets:values.extend([b['position'][0]/192,b['position'][1]/448,b['velocity_raw'][0]/4.5,b['velocity_raw'][1]/4.5])
        return {'state':np.array(values,dtype=np.float32),'bomb_clock':np.zeros(1,dtype=np.float32)}

"""Adopted offline death scale; observation and physics remain unchanged."""
from touhou_ai.spatial_input_candidates import CandidateEnv

class ScaledCandidateEnv(CandidateEnv):
    def __init__(self, variant='action_grid', death_reward=-1):
        if death_reward>=0:raise ValueError('Death reward must be negative')
        self.death_reward=float(death_reward)
        super().__init__(variant)
    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        return obs,reward*(self.death_reward/-60),done,truncated,info

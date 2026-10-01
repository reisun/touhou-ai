"""Observed-wall threat rewards, one threat budget per synthetic episode."""
import numpy as np
from touhou_ai.side_dodge import SideGrid,SideNumerical
from touhou_ai.fixed_dodge import DIRECTIONS
class AvoidanceRewards:
 def reset(self,**kwargs):
  self.threat_seen=False;self.risk_paid=False;self.pass_paid=False;self.reward_totals={'hit':0.,'risk':0.,'pass':0.}
  return super().reset(**kwargs)
 def step(self,action):
  pos=self.pos.copy();bullets=np.array([b['position'] for b in self.bullets]);velocity=DIRECTIONS[int(action[0])]*(2 if action[2] else 4.5)
  # Current constant-velocity observations only. Identical subframe AABB geometry.
  def collides(v):
   t=np.arange(0,16.01,.25)[:,None]
   p=np.clip(pos+v*np.minimum(t,2.),[-184,32],[184,432]);bs=bullets[None,:,:]+(np.array([0.,3.])*t)[:,None,:]
   return bool(np.any(np.all(np.abs(bs-p[:,None,:])<=5,axis=2)))
  threatened=collides(np.zeros(2))
  if threatened:self.threat_seen=True
  improved=threatened and not collides(velocity)
  obs,base,done,trunc,info=super().step(action);risk=0.;passed=0.
  if not self.dead:
   if improved and not self.risk_paid:risk=2.;self.risk_paid=True
   # Whole tracked wall has actually passed player, not merely predicted clear.
   if self.threat_seen and not self.pass_paid and min(b['position'][1] for b in self.bullets)>self.pos[1]+5:
    passed=4.;self.pass_paid=True
  self.reward_totals['hit']+=base;self.reward_totals['risk']+=risk;self.reward_totals['pass']+=passed
  info['reward_components']={'hit':base,'risk':risk,'pass':passed}
  if done:info['reward_totals']=self.reward_totals.copy()
  return obs,base+risk+passed,done,trunc,info
class RewardGrid(AvoidanceRewards,SideGrid):pass
class RewardNumerical(AvoidanceRewards,SideNumerical):pass

"""Narrow corridor, two alternating moving gaps; 44 observed entities."""
import numpy as np
from touhou_ai.fixed_dodge import FixedDodge,DIRECTIONS
from touhou_ai.numerical_dodge import NumericalDodge
class Corridor:
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed)
  if seed is not None or not hasattr(self,'side_rng'):self.side_rng=np.random.default_rng(7 if seed is None else seed)
  self.side=(options or {}).get('side')
  if self.side is None:self.side=int(self.side_rng.choice([-1,1]))
  if self.side not in [-1,1]:raise ValueError('invalid side')
  def bullet(x,y,v,size):return {'position':[float(x),float(y)],'velocity_raw':[0.,float(v)],'hitbox_raw':size,'flags_raw':2}
  self.bullets=[bullet(x,y,0,[8.,56.]) for x in [-24,24] for y in range(32,425,56)]
  for y,side in [(210,self.side),(90,-self.side)]:
   self.bullets.extend(bullet(x,y,3,[8.,8.]) for x in range(-60,61,8) if abs(x-side*8)>4)
  self.bullets.sort(key=lambda b:(b['velocity_raw'][1],b['position'][1],b['position'][0]))
  assert len(self.bullets)==44
  return self.observe(),{}
 def opposite_observation(self):
  saved=self.bullets;self.bullets=sorted([{**b,'position':[-b['position'][0],b['position'][1]]} for b in saved],key=lambda b:(b['velocity_raw'][1],b['position'][1],b['position'][0]))
  try:return self.observe()
  finally:self.bullets=saved
 def step(self,action):
  if action[1] or action[3]:raise ValueError('shot/bomb disabled')
  self.focus=int(action[2]);v=DIRECTIONS[int(action[0])]*(2 if self.focus else 4.5)
  positions=np.array([b['position'] for b in self.bullets]);vel=np.array([b['velocity_raw'] for b in self.bullets]);extent=np.array([b['hitbox_raw'] for b in self.bullets])*.5+1
  for _ in range(8):
   self.pos=np.clip(self.pos+v*.25,[-184,32],[184,432]);positions+=vel*.25
   if np.any(np.all(np.abs(positions-self.pos)<=extent,axis=1)):self.dead=True;break
  for b,p in zip(self.bullets,positions):b['position']=p.tolist()
  self.frame+=2;done=self.dead or self.frame>=160
  return self.observe(),-60. if self.dead else 0.,done,False,{'success':done and not self.dead}
class CorridorGrid(Corridor,FixedDodge):pass
class CorridorNumerical(Corridor,NumericalDodge):
 def observe(self):
  o=super().observe();o['state'][2:4]=0
  return o

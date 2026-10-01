"""Source-derived Normal Autumn Sky, first 600F after first emission.
Not a bit-exact TH10 emulator. See docs/autumn-sky.md for fidelity boundaries.
"""
import gymnasium as gym
import numpy as np
from touhou_ai.dual_grid import DualGridContract
from touhou_ai.fixed_dodge import DIRECTIONS

COUNT=38
HORIZON=600
MAX_BULLETS=COUNT*4*5
TURNS=np.array([[np.pi/2,-np.pi/2,np.pi/2],[-np.pi/2,np.pi/2,-np.pi/3],
 [np.pi/2,-np.pi/2,np.pi/3.5],[-np.pi/2,np.pi/2,-np.pi/2.5],
 [-np.pi/2,np.pi/2,-np.pi/2],[np.pi/2,-np.pi/2,np.pi/3],
 [-np.pi/2,np.pi/2,-np.pi/3.5],[np.pi/2,-np.pi/2,np.pi/2.5]])

class AutumnSky(DualGridContract):
 horizon=HORIZON
 def reset(self,*,seed=None,options=None):
  if seed is not None or not hasattr(self,'rng'):self.rng=np.random.default_rng(7 if seed is None else seed)
  self.pos=np.array([0.,330.]);self.boss=np.array([0.,128.]);self.frame=0;self.focus=0;self.dead=False
  self.xy=np.zeros((MAX_BULLETS,2));self.vel=np.zeros_like(self.xy)
  self.angle=np.zeros(MAX_BULLETS);self.birth=np.zeros(MAX_BULLETS,dtype=int)
  self.group=np.zeros(MAX_BULLETS,dtype=int);self.alive=np.zeros(MAX_BULLETS,dtype=bool);self.n=0
  self.move_angle=0.;self.emissions=[]
  self._emit()
  return self.observe(),{}
 def _emit(self):
  phase=float(self.rng.uniform(-np.pi,np.pi));groups=np.repeat(np.arange(4)+(4 if (self.frame//120)%2 else 0),COUNT)
  angles=np.tile(np.arange(COUNT)*2*np.pi/COUNT+phase,4);sl=slice(self.n,self.n+COUNT*4)
  self.angle[sl]=angles;self.group[sl]=groups;self.birth[sl]=self.frame;self.alive[sl]=True
  uv=np.column_stack((np.cos(angles),np.sin(angles)))
  self.xy[sl]=self.boss-8*uv;self.vel[sl]=uv  # spawn animation: half of speed 2
  self.n+=COUNT*4
  self.emissions.append({'frame':self.frame,'boss':self.boss.tolist(),'angle':phase,'count':152})
  # ECL moveRand(60,4,2): signed random direction biased by bounds/player x.
  r=float(self.rng.uniform(-1,1));x,y=self.boss
  if x < -70:a=r*np.pi/3
  elif x>70:a=r*np.pi/3+np.pi
  elif x<self.pos[0]:a=r*np.pi/2
  else:a=r*2*np.pi+np.pi
  a=(a+np.pi)%(2*np.pi)-np.pi
  if y<71:a=abs(a)
  elif y>121:a=-abs(a)
  self.move_angle=a
 def _frame_velocity(self):
  ids=np.flatnonzero(self.alive);age=self.frame-self.birth[ids];angle=self.angle[ids].copy();speed=np.ones(len(ids))
  for k,turn_age in enumerate((75,136,197)):
   angle+=np.where(age>=turn_age,TURNS[self.group[ids],k],0.)
  for start in (15,76,137):
   mask=(age>=start)&(age<start+60);speed[mask]=2*(1-(age[mask]-start)/60)
  speed[(age==75)|(age==136)]=2
  speed[age>=197]=1.2
  speed[age==15]+=1 # half-speed spawn update falls through into first active update
  self.vel[ids]=np.column_stack((np.cos(angle),np.sin(angle)))*speed[:,None]
  return ids,age>=15
 def _cull(self):
  age=self.frame-self.birth
  half_x=np.where(age<15,15.,8.);half_y=np.where(age<15,15.,7.)
  outside=(self.xy[:,0]+half_x<=-192)|(self.xy[:,0]-half_x>=192)|(self.xy[:,1]+half_y<=-64)|(self.xy[:,1]-half_y>=448)
  self.alive[(age>=10)&outside]=False
 def step(self,action):
  if action[1] or action[3]:raise ValueError('shot/bomb disabled')
  if self.dead or self.frame>=self.horizon:raise RuntimeError('reset required')
  self.focus=int(action[2]);pv=DIRECTIONS[int(action[0])]*(2 if self.focus else 4.5)
  for _ in range(2):
   if self.frame and self.frame%120==0:self._emit()
   ids,collidable=self._frame_velocity()
   for _ in range(4):
    self.pos=np.clip(self.pos+pv*.25,[-184,32],[184,432]);self.xy[ids]+=self.vel[ids]*.25
    if np.any(np.all(np.abs(self.xy[ids[collidable]]-self.pos)<=3,axis=1)):self.dead=True;break
   # Original screen boundary, not local observation-window boundary.
   self._cull()
   t=self.frame%120
   if t<60:
    self.boss+=np.array([np.cos(self.move_angle),np.sin(self.move_angle)])*2*(1-t/60)**2
    self.boss=np.clip(self.boss,[-140,46],[140,146])
   self.frame+=1
   if self.dead:break
  done=self.dead or self.frame>=self.horizon
  return self.observe(),-60. if self.dead else 0.,done,False,{'success':done and not self.dead,'frames':self.frame}
 def observe(self):
  ids=np.flatnonzero(self.alive)
  raw={'stage':1,'stage_frame':self.frame,'lives_raw':0,'power_raw':0,
   'player':{'position':self.pos.tolist(),'velocity_raw':[0,0],'hitbox_raw':[1,1],'status':4 if self.dead else 1,'invincibility_raw':0,'focus_raw':self.focus},
   'bullets':[{'position':p.tolist(),'velocity_raw':v.tolist(),'hitbox_raw':[4.,4.],'flags_raw':2} for p,v in zip(self.xy[ids],self.vel[ids])],
   'enemies':[],'items':[],'lasers':[],'player_shots':[],'bomb':{'state':0},'spell':None}
  return super().encode(raw)

class AutumnNumerical(AutumnSky):
 def __init__(self):
  super().__init__()
  self.observation_space=gym.spaces.Dict({'state':gym.spaces.Box(-2,2,(5+MAX_BULLETS*5,),dtype=np.float32),'bomb_clock':gym.spaces.Box(0,1,(1,),dtype=np.float32)})
 def observe(self):
  state=np.zeros(5+MAX_BULLETS*5,dtype=np.float32);state[:5]=[self.pos[0]/192,self.pos[1]/448,0,0,self.focus]
  slots=state[5:].reshape(-1,5);ids=np.flatnonzero(self.alive)
  slots[ids,:2]=self.xy[ids]/[192,448];slots[ids,2:4]=self.vel[ids]/4.5;slots[ids,4]=1
  return {'state':state,'bomb_clock':np.zeros(1,dtype=np.float32)}

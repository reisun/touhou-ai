"""Receding-horizon reference controller using current positions/velocities only."""
import numpy as np
from touhou_ai.fixed_dodge import DIRECTIONS
ACTIONS=np.array([[d,0,f,0] for f in [1,0] for d in range(9)])
VELOCITIES=np.concatenate((DIRECTIONS*2,DIRECTIONS*4.5))
def choose(pos,xy,vel,sigma=8.,allow_fast=True):
 # Exclude distant bullets that cannot approach within 16 frames.
 keep=np.linalg.norm(xy-pos,axis=1)<=100
 xy=xy[keep];vel=vel[keep]
 if not len(xy):return ACTIONS[0].copy()
 times=np.array([2.,4.,8.,12.,16.])
 player=np.clip(pos+VELOCITIES[:,None,:]*times[None,:,None],[-184,32],[184,432])
 bullets=xy[None,:,:]+vel[None,:,:]*times[:,None,None]
 delta=bullets[None,:,:,:]-player[:,:,None,:]
 distance=np.maximum(np.abs(delta)-3.,0.)
 risk=np.exp(-np.sum(distance**2,axis=-1)/(2*sigma*sigma)).sum(axis=2)
 costs=(risk*np.array([1.,1.,.8,.6,.4])).sum(axis=1)+np.linalg.norm(VELOCITIES,axis=1)*.015
 if not allow_fast:costs[9:]=np.inf
 return ACTIONS[int(np.argmin(costs))].copy()

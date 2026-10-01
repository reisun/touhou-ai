"""Offline single-factor refinements of the adopted action-grid baseline."""
import functools
import numpy as np
from touhou_ai.spatial_input_candidates import CandidateEnv, TILES, LOW, HIGH, bullet_arrays
from touhou_ai.fixed_dodge import DIRECTIONS
from touhou_ai.autumn_training import build_spatial_reference_model as build_model

def forecast(raw, continuous=False, graded=False, prune=True):
    if raw['lasers']: raise ValueError('Offline bullet-only experiment')
    origin=np.asarray(raw['player']['position'],float)
    p,v,size=bullet_arrays(raw)
    extent=size/2+np.asarray(raw['player']['hitbox_raw'])
    if prune:
        # Conservative coordinate-wise bound on relative displacement.
        # Include initial clipping displacement even for diagnostic out-of-bounds states.
        reach=4.5*(16 if continuous else 2)+np.abs(np.clip(origin,LOW,HIGH)-origin)
        near=np.all(np.abs(p-origin)<=np.abs(v)*16+extent+reach,axis=1)
        p,v,extent=p[near],v[near],extent[near]
    out=np.zeros((4,3,3),np.float32)
    if not len(p): return out
    for focus,speed in enumerate((4.5,2.)):
        for direction,(iy,ix) in enumerate(TILES):
            velocity=DIRECTIONS[direction]*speed
            stop=16 if continuous else 2
            times=[0.,2.,16.]
            for axis in (0,1):
                if velocity[axis]:
                    t=((HIGH if velocity[axis]>0 else LOW)[axis]-origin[axis])/velocity[axis]
                    if 0<t<stop: times.append(float(t))
            first=float('inf')
            times=sorted(set(times))
            for a,b in zip(times[:-1],times[1:]):
                x0=np.clip(origin+min(a,stop)*velocity,LOW,HIGH)
                x1=np.clip(origin+min(b,stop)*velocity,LOW,HIGH)
                rel=p+a*v-x0; delta=(b-a)*v-(x1-x0)
                stationary=np.abs(delta)<1e-12
                den=np.where(stationary,1,delta)
                u0=(-extent-rel)/den;u1=(extent-rel)/den
                enter=np.maximum(np.where(stationary,-np.inf,np.minimum(u0,u1)).max(axis=1),0)
                leave=np.minimum(np.where(stationary,np.inf,np.maximum(u0,u1)).min(axis=1),1)
                valid=~np.any(stationary&(np.abs(rel)>extent),axis=1)
                hit=valid&(enter<=leave)
                if hit.any(): first=min(first,float(a+(b-a)*enter[hit].min()))
            for channel,horizon in ((focus,2),(2+focus,16)):
                if first<=horizon:
                    # Nonzero at the horizon, one at immediate contact.
                    out[channel,iy,ix]=1-first/(horizon+1) if graded else 1
    return out

class RefinedEnv(CandidateEnv):
    def __init__(self, refinement='control'):
        self.refinement=refinement
        super().__init__('action_grid')
    def observe(self):
        if self.refinement in ('urgency','continuous'):
            obs=self.encoder.base.encode(self.raw_observation())
            obs['action_grid']=forecast(self.raw_observation(),continuous=self.refinement=='continuous',graded=self.refinement=='urgency').astype(np.float16).astype(np.float32)
            return obs
        return super().observe()
    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        if self.refinement=='scale': reward/=60
        return obs,reward,done,truncated,info

def build(refinement,seed):
    model,_,config=build_model('cnn',seed)
    cls=functools.partial(RefinedEnv,refinement=refinement)
    model.set_env(cls())
    return model,cls,config

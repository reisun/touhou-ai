"""Offline speed and numerical-coordinate ablations; physics unchanged."""
import numpy as np
import torch
from torch.distributions import Categorical
from touhou_ai.autumn_sky import AutumnSky, AutumnNumerical
from touhou_ai.evasion_only import EvasionPolicy

class SpeedPolicy(EvasionPolicy):
 def __init__(self,*args,fixed_focus=None,**kwargs):
  self.fixed_focus=fixed_focus
  super().__init__(*args,**kwargs)
 def _scheduled(self,latent,obs):
  dist=super()._scheduled(latent,obs)
  if self.fixed_focus is not None:
   logits=torch.full((latent.shape[0],2),-1e9,device=latent.device,dtype=latent.dtype)
   logits[:,self.fixed_focus]=0
   dist.distribution[2]=Categorical(logits=logits)
  return dist

class SpeedMixin:
 def __init__(self,*,fixed_focus=None,relative=False):
  self.fixed_focus=fixed_focus;self.relative=relative
  super().__init__()
 def reset(self,**kwargs):
  super().reset(**kwargs)
  if self.fixed_focus is not None:self.focus=self.fixed_focus
  return self.observe(),{}
 def step(self,action):
  if self.fixed_focus is not None and int(action[2])!=self.fixed_focus:
   raise ValueError('policy/environment focus mismatch')
  return super().step(action)

class GridAblation(SpeedMixin,AutumnSky):pass
class NumericalAblation(SpeedMixin,AutumnNumerical):
 def observe(self):
  obs=super().observe()
  if self.relative:
   slots=obs['state'][5:].reshape(-1,5);active=slots[:,4]>0
   slots[active,:2]-=np.asarray(self.pos/[192,448],dtype=np.float32)
  return obs

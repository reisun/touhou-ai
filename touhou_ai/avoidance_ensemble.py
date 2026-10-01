"""Mean joint move/focus probabilities; neural policies only, no rule controller."""
import numpy as np
import torch
from touhou_ai.simulation_speed import distributions
class Ensemble:
 def __init__(self,models):
  if not models:raise ValueError('models required')
  self.models=models
 def __call__(self,obs):
  joint=[]
  with torch.inference_mode():
   for m in self.models:
    ds=distributions(m.policy,obs)
    joint.append(ds[2].probs[:,:,None]*ds[0].probs[:,None,:])
   indices=torch.stack(joint).mean(0).flatten(1).argmax(1).cpu().numpy()
  actions=np.zeros((len(indices),4),dtype=np.int64);actions[:,0]=indices%9;actions[:,2]=indices//9
  return actions

"""Horizontal reflection of observed grids and matching movement labels."""
import numpy as np
MIRROR=np.array([0,1,8,7,6,5,4,3,2])
def reflect_batch(obs,actions,indices):
 for key in ['local_grid','global_grid']:
  obs[key][indices]=obs[key][indices,...,::-1]
 obs['player'][indices,0]*=-1
 obs['player'][indices,2]*=-1
 actions[indices,0]=MIRROR[actions[indices,0]]

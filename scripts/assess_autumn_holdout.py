"""Held-out evaluation: environment seeds5000..5095, RNG19000..19095."""
import numpy as np
import torch
from touhou_ai.simulation_speed import distributions

def assess_holdout(model,cls,n=96,mode='sample'):
 rows=[]
 for offset in range(0,n,8):
  envs=[cls() for _ in range(min(8,n-offset))]
  obs=[e.reset(seed=5000+offset+i)[0] for i,e in enumerate(envs)]
  initial_obs=obs.copy()
  frozen_probs=None
  if mode.startswith('frozen_'):
   batch={k:np.stack([o[k] for o in obs]) for k in obs[0]}
   with torch.inference_mode():frozen_probs=[d.probs.cpu().numpy() for d in distributions(model.policy,batch)]
   for i,e in enumerate(envs):e.unwrapped.observe=lambda initial=obs[i]:initial
  rngs=[np.random.default_rng(19000+offset+i) for i in range(len(envs))];active=list(range(len(envs)));initial=None
  for t in range(300):
   if frozen_probs is not None:
    probs=[p[active] for p in frozen_probs]
   elif model is not None:
    selected=initial_obs if mode.startswith('frozen_') else obs
    batch={k:np.stack([selected[i][k] for i in active]) for k in obs[0]}
    with torch.inference_mode():probs=[d.probs.cpu().numpy() for d in distributions(model.policy,batch)]
   following=[]
   for j,i in enumerate(active):
    if model is None:a=[0,0,0,0] if mode=='stop' else [int(rngs[i].integers(9)),0,int(rngs[i].integers(2)),0]
    elif mode in ['greedy','frozen_greedy']:a=[int(p[j].argmax()) for p in probs]
    else:
     u=rngs[i].random(4);a=[min(int(np.searchsorted(np.cumsum(p[j]),u[h])),p.shape[1]-1) for h,p in enumerate(probs)]
    obs[i],r,d,_,info=envs[i].step(a)
    if d:rows.append({'seed':5000+offset+i,'success':info['success'],'frames':info['frames'],'end_xy':envs[i].pos.tolist()})
    else:following.append(i)
   active=following
   if not active:break
  if active:raise AssertionError('assessment truncated before environment termination')
 return {'n':n,'survival':sum(r['success'] for r in rows)/n,'mean_frames':float(np.mean([r['frames'] for r in rows])),'episodes':rows}

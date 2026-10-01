import sys,pathlib,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_two_sets import TwoSetsGrid,TwoSetsNumerical
from touhou_ai.simulation_speed import distributions

def assess(model,cls,mode,n=24,seed_start=1000,action_seed_start=9000,action_provider=None):
 start=time.perf_counter();rows=[]
 for offset in range(0,n,8):
  envs=[cls() for _ in range(min(8,n-offset))];obs=[e.reset(seed=seed_start+offset+i)[0] for i,e in enumerate(envs)];initial=obs.copy()
  rngs=[np.random.default_rng(action_seed_start+offset+i) for i in range(len(envs))];active=list(range(len(envs)))
  for t in range(900):
   if model is not None or action_provider is not None:
    selected=initial if mode.startswith('frozen') else obs
    batch={k:np.stack([selected[i][k] for i in active]) for k in obs[0]}
    if mode.startswith('masked'):
     if 'local_grid' in batch:
      batch['local_grid'][:,1:4]=0;batch['global_grid'][:,:3]=0
     else:batch['state'][:,5:]=0
    with torch.inference_mode():
     if action_provider is not None:provided=action_provider(batch)
     else:probs=[d.probs.cpu().numpy() for d in distributions(model.policy,batch)]
   following=[]
   for j,i in enumerate(active):
    if action_provider is not None:a=provided[j]
    elif model is None:
     a=[6,0,0,0] if mode=='down_left' else [0,0,0,0] if mode=='stop' else [int(rngs[i].integers(9)),0,int(rngs[i].integers(2)),0]
    elif mode.endswith('greedy'):a=[int(p[j].argmax()) for p in probs]
    else:
     u=rngs[i].random(4);a=[min(int(np.searchsorted(np.cumsum(p[j]),u[h])),p.shape[1]-1) for h,p in enumerate(probs)]
    obs[i],_,done,trunc,info=envs[i].step(a)
    if done or trunc:rows.append({'seed':seed_start+offset+i,**info})
    else:following.append(i)
   active=following
   if not active:break
  if active:raise AssertionError('unfinished evaluation')
 elapsed=time.perf_counter()-start;frames=sum(r['frames'] for r in rows)
 return {'n':n,'one_set':sum(r['sets_cleared']>=1 for r in rows)/n,'two_sets':sum(r['success'] for r in rows)/n,'timeouts':sum(r['timeout'] for r in rows),'mean_seconds':frames/n/60,'wall_seconds':elapsed,'aggregate_realtime_factor':frames/60/elapsed,'episodes':rows}

def main():
 torch.set_num_threads(1);source=pathlib.Path('artifacts/autumn-sky-20260928-190412');out=pathlib.Path('artifacts/autumn-two-sets-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();rows=[];print(out,flush=True)
 jobs=[('baseline',None,TwoSetsNumerical,['stop','random','down_left'])]
 for name,cls in [('numerical',TwoSetsNumerical),('narrow-cnn',TwoSetsGrid)]:
  jobs.append((name,PPO.load(source/(name+'-16384.zip'),device='cpu'),cls,['sample','greedy','frozen_sample','frozen_greedy']))
 for name,m,cls,modes in jobs:
  for mode in modes:
   row={'model':name,'mode':mode,**assess(m,cls,mode)};rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='episodes'}),flush=True)
 print('complete',out,flush=True)
if __name__=='__main__':main()

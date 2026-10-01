"""Source-derived Autumn Sky diagnostic; independent of the live game."""
import sys,pathlib,time,json,argparse,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_sky import AutumnSky,AutumnNumerical
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.simulation_speed import distributions,tune_cpu
from scripts.continue_dodge_comparison import Progress

def assess(model,cls,n=24,mode='sample'):
 rows=[]
 for offset in range(0,n,8):
  envs=[cls() for _ in range(min(8,n-offset))]
  obs=[e.reset(seed=1000+offset+i)[0] for i,e in enumerate(envs)]
  initial_obs=obs.copy()
  rngs=[np.random.default_rng(9000+offset+i) for i in range(len(envs))];active=list(range(len(envs)));initial=None
  for t in range(300):
   if model is not None:
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
    if d:rows.append({'seed':1000+offset+i,'success':info['success'],'frames':info['frames'],'end_xy':envs[i].pos.tolist()})
    else:following.append(i)
   active=following
   if not active:break
  if active:raise AssertionError('assessment truncated before environment termination')
 return {'n':n,'survival':sum(r['success'] for r in rows)/n,'mean_frames':float(np.mean([r['frames'] for r in rows])),'episodes':rows}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=16384);args=ap.parse_args()
 out=pathlib.Path('artifacts/autumn-sky-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();print(out,flush=True)
 (out/'manifest.json').write_text(json.dumps({'environment':'autumn-sky-source-v2','physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'horizon_frames':600,'training_seed':7,'evaluation_seeds':list(range(1000,1024)),'rewards':{'death':-60},'steps':args.steps,'fidelity':'source-derived; no live trajectory parity'},indent=2))
 rows=[];baselines={k:assess(None,AutumnNumerical,mode=k) for k in ['stop','random']};(out/'baselines.json').write_text(json.dumps(baselines,indent=2))
 print('baselines',json.dumps({k:{a:v[a] for a in ['n','survival','mean_frames']} for k,v in baselines.items()}),flush=True)
 for name,cls in [('numerical',AutumnNumerical),('narrow-cnn',AutumnSky)]:
  kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
  if name=='narrow-cnn':kw['features_extractor_class']=NarrowGridFeatures
  m=PPO(EvasionPolicy,cls(),seed=7,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0)
  tune_cpu(m);start=time.perf_counter()
  for target in [0,args.steps]:
   if target:m.learn(total_timesteps=target,reset_num_timesteps=False,callback=Progress())
   ts=torch.get_rng_state();ns=np.random.get_state()
   row={'model':name,'steps':m.num_timesteps,'parameters':sum(p.numel() for p in m.policy.parameters()),'sample':assess(m,cls),'greedy':assess(m,cls,mode='greedy'),'seconds':time.perf_counter()-start}
   torch.set_rng_state(ts);np.random.set_state(ns);rows.append(row);m.save(out/(name+'-'+str(target)));(out/'results.json').write_text(json.dumps(rows,indent=2))
   print(json.dumps({k:({a:v[a] for a in ['n','survival','mean_frames']} if k in ['sample','greedy'] else v) for k,v in row.items()}),flush=True)
 print('complete',out,flush=True)
if __name__=='__main__':main()

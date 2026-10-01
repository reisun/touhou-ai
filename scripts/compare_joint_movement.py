import sys,pathlib,json,time,hashlib,functools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import GridAblation,NumericalAblation
from touhou_ai.joint_movement import JointMovementPolicy
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu,distributions
from scripts.continue_dodge_comparison import Progress

def assess_joint(model,cls,n=24,mode='sample'):
 rows=[]
 for offset in range(0,n,8):
  envs=[cls() for _ in range(min(8,n-offset))]
  obs=[e.reset(seed=1000+offset+i)[0] for i,e in enumerate(envs)]
  initial_obs=obs.copy();switches=[0]*len(envs);focus_counts=[0]*len(envs);decisions=[0]*len(envs);last_focus=[0]*len(envs)
  rngs=[np.random.default_rng(9000+offset+i) for i in range(len(envs))];active=list(range(len(envs)));initial=None
  for t in range(300):
   if model is not None:
    selected=initial_obs if mode.startswith('frozen_') else obs
    batch={k:np.stack([selected[i][k] for i in active]) for k in obs[0]}
    with torch.inference_mode():probs=[d.probs.cpu().numpy() for d in distributions(model.policy,batch)]
   following=[]
   for j,i in enumerate(active):
    if model is None:a=[0,0,0,0] if mode=='stop' else [int(rngs[i].integers(9)),0,int(rngs[i].integers(2)),0]
    else:
     if len(probs)!=1:raise ValueError('joint evaluator requires one 18-way categorical')
     p=probs[0][j].reshape(9,2)
     if mode in ['greedy','frozen_greedy']:
      idx=int(p.argmax());a=[idx//2,0,idx%2,0]
     else:
      # Sample the joint via exact marginal/conditional factors, reusing baseline u0/u2.
      u=rngs[i].random(4);marginal=p.sum(axis=1)
      direction=min(int(np.searchsorted(np.cumsum(marginal),u[0])),8)
      conditional=p[direction]/marginal[direction]
      focus=min(int(np.searchsorted(np.cumsum(conditional),u[2])),1)
      a=[direction,0,focus,0]
    decisions[i]+=1;focus_counts[i]+=a[2];switches[i]+=int(a[2]!=last_focus[i]);last_focus[i]=a[2]
    obs[i],r,d,_,info=envs[i].step(a)
    if d:rows.append({'seed':1000+offset+i,'success':info['success'],'frames':info['frames'],'end_xy':envs[i].pos.tolist(),'focus_fraction':focus_counts[i]/decisions[i],'switches_per_100_actions':100*switches[i]/decisions[i]})
    else:following.append(i)
   active=following
   if not active:break
  if active:raise AssertionError('assessment truncated before environment termination')
 return {'n':n,'survival':sum(r['success'] for r in rows)/n,'mean_frames':float(np.mean([r['frames'] for r in rows])),'episodes':rows,'focus_fraction':float(np.mean([r['focus_fraction'] for r in rows])),'switches_per_100_actions':float(np.mean([r['switches_per_100_actions'] for r in rows]))}

def main():
 rep=sys.argv[2]
 if rep not in ['numerical','cnn']:raise ValueError(rep)
 out=pathlib.Path(sys.argv[1])/rep;out.mkdir(parents=True,exist_ok=True)
 rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
 seeds=[7,17,27] if rep=='numerical' else [7]
 baseline=json.loads(pathlib.Path('artifacts/autumn-ablation-20260928/results.json').read_text())
 baseline=[r for r in baseline if r['speed']=='switch' and r['representation']==('relative' if rep=='numerical' else 'cnn') and r['seed'] in seeds]
 assert len(baseline)==len(seeds)
 (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
 manifest={'representation':rep,'seeds':seeds,'steps':16384,'reward':{'death':-60},'action':'joint 18 categorical, index=direction*2+focus, shot/bomb disabled','physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'evaluation':'24 seeds1000..1023; joint sampled by direction marginal and focus conditional with same per-episode uniforms as baseline','common_initial_weights':'encoder,actor MLP,critic identical to independent-head baseline for same seed; tested','baseline_source':'artifacts/autumn-ablation-20260928','fidelity':'source-derived Autumn Sky first600F, no live trajectory parity'}
 assert manifest['physics_sha256']==json.loads(pathlib.Path('artifacts/autumn-ablation-20260928/manifest.json').read_text())['physics_sha256']
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 cls=functools.partial(NumericalAblation,relative=True) if rep=='numerical' else GridAblation
 for seed in seeds:
  if any(r['seed']==seed for r in rows):continue
  print('START',rep,seed,flush=True);start=time.perf_counter()
  kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
  if rep=='cnn':kw['features_extractor_class']=NarrowGridFeatures
  m=PPO(JointMovementPolicy,cls(),seed=seed,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0)
  tune_cpu(m);ts=torch.get_rng_state();ns=np.random.get_state();before=assess_joint(m,cls)
  torch.set_rng_state(ts);np.random.set_state(ns)
  m.learn(total_timesteps=16384,callback=Progress());m.save(out/f'joint-{seed}')
  evaluations={mode:assess_joint(m,cls,mode=mode) for mode in ['sample','greedy','frozen_sample','frozen_greedy']}
  row={'representation':rep,'seed':seed,'before':before,'evaluations':evaluations,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in m.policy.parameters())}
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
  print('RESULT',rep,seed,json.dumps({k:{'survival':v['survival'],'seconds':v['mean_frames']/60,'focus_fraction':v['focus_fraction'],'switches_per_100_actions':v['switches_per_100_actions']} for k,v in evaluations.items()}),flush=True)
 (out/'status.json').write_text(json.dumps({'status':'complete','seeds':seeds}))
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()

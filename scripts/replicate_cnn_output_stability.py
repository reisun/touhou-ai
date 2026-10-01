"""Fixed-sample CNN stability replication; matched independent/joint output pairs."""
import sys,json,pathlib,time,hashlib
import numpy as np,torch
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import SpeedPolicy,GridAblation
from touhou_ai.joint_movement import JointMovementPolicy
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu,distributions
from scripts.compare_joint_adopted import decode_joint
from scripts.continue_dodge_comparison import Progress
OUT=pathlib.Path('artifacts/cnn-output-stability-20260929')
FRESH=[37,47,57,67,77,87,97]
OLD=[7,17,27]
JOBS=[(s,v) for s in FRESH+OLD for v in ['independent','joint']]

def evaluate(model,variant,start=0,n=96):
 model.policy.set_training_mode(False);rows=[]
 for offset in range(start,start+n,8):
  envs=[GridAblation() for _ in range(min(8,start+n-offset))]
  obs=[e.reset(seed=5000+offset+i)[0] for i,e in enumerate(envs)]
  rng=[np.random.default_rng(19000+offset+i) for i in range(len(envs))];active=list(range(len(envs)))
  for step in range(300):
   with torch.inference_mode():ds=distributions(model.policy,{k:np.stack([obs[i][k] for i in active]) for k in obs[0]})
   probs=[d.probs.numpy() for d in ds];assert len(probs)==(1 if variant=='joint' else 4)
   following=[]
   for j,i in enumerate(active):
    u=rng[i].random(4)
    a=decode_joint(probs[0][j],u) if variant=='joint' else [min(int(np.searchsorted(np.cumsum(p[j]),u[h])),p.shape[1]-1) for h,p in enumerate(probs)]
    obs[i],_,done,_,info=envs[i].step(a)
    if done:rows.append({'seed':5000+offset+i,'success':bool(info['success']),'frames':info['frames']})
    else:following.append(i)
   active=following
   if not active:break
  if active:raise AssertionError('unfinished evaluation')
 return sorted(rows,key=lambda r:r['seed'])

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 config=json.loads(pathlib.Path('configs/autumn-learning-reference-v2.json').read_text())
 manifest={'fresh_seeds':FRESH,'exploratory_seeds':OLD,'jobs':JOBS,'steps':16384,'evaluation_episodes':192,'environment_seeds':[5000,5191],'action_rng_seeds':[19000,19191],'config':config,'physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'primary':'fresh7 paired training seeds: between-seed SD of stochastic survival; paired two-axis bootstrap for SD ratio joint/independent; mean performance and low-tail secondary','secondary':'all10 seeds including original discovery sample, and mean survival time; separate panels96+96','fixed_stopping':'complete all preselected jobs, do not stop for a favorable result','scope':'stability across training seeds at16384 decisions, not stability over training updates or live-game generalization','adoption_criterion':'upper95% bootstrap SD ratio<1 on fresh seeds AND lower95% mean survival difference>-0.02; otherwise retain current default pending evidence'}
 path=OUT/'manifest.json'
 if path.exists():assert json.loads(path.read_text())==manifest
 else:path.write_text(json.dumps(manifest,indent=2))

def main():
 if sys.argv[1]=='prepare':prepare();return
 manifest=json.loads((OUT/'manifest.json').read_text());config=manifest['config'];assert config['representations']['cnn']['algorithm']=='SeparateClipPPO'
 worker=sys.argv[1]
 jobs=[(s,v) for s in OLD for v in ['independent','joint']] if worker=='old' else JOBS[int(worker)::4]
 for seed,variant in jobs:
  out=OUT/str(seed)/variant;out.mkdir(parents=True,exist_ok=True)
  if (out/'result.json').exists():continue
  torch.set_num_threads(1);began=time.perf_counter();print('START',seed,variant,flush=True)
  if seed in OLD:
   path=pathlib.Path(f'artifacts/autumn-joint-adopted-20260929/cnn/{seed}/joint.zip') if variant=='joint' else pathlib.Path(f'artifacts/autumn-separate-clip-20260928/cnn-switch-{seed}.zip')
   m=SeparateClipPPO.load(path,device='cpu')
  elif (out/'model.zip').exists():m=SeparateClipPPO.load(out/'model',device='cpu')
  else:
   kw={'share_features_extractor':False,'features_extractor_class':NarrowGridFeatures,'net_arch':{'pi':[256,128],'vf':[256,128]}}
   m=SeparateClipPPO(JointMovementPolicy if variant=='joint' else SpeedPolicy,GridAblation(),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0,**config['ppo']);tune_cpu(m)
   m.learn(16384,callback=Progress());m.save(out/'model')
  assert m.num_timesteps==16384
  panels=[]
  for start in [0,96]:
   file=out/f'panel-{start}.json'
   if file.exists():episodes=json.loads(file.read_text())
   elif seed in OLD and start==0:
    if variant=='joint':row=json.loads(pathlib.Path(f'artifacts/autumn-joint-adopted-20260929/cnn/{seed}/result.json').read_text())
    else:row=next(r for r in json.loads(pathlib.Path('artifacts/autumn-separate-clip-20260928/holdout.json').read_text()) if r['representation']=='cnn' and r['variant']=='separate' and r['seed']==seed)
    episodes=sorted([{k:e[k] for k in ['seed','success','frames']} for e in row['evaluations']['sample']['episodes']],key=lambda e:e['seed'])
   else:episodes=evaluate(m,variant,start=start)
   file.write_text(json.dumps(episodes));panels.extend(episodes)
   print('EVAL',seed,variant,start,round(np.mean([e['success'] for e in episodes])*100,2),flush=True)
  assert len(panels)==192 and len({e['seed'] for e in panels})==192
  result={'seed':seed,'variant':variant,'fresh':seed in FRESH,'steps':m.num_timesteps,'survival':float(np.mean([e['success'] for e in panels])),'seconds_alive':float(np.mean([e['frames'] for e in panels])/60),'wall_seconds':time.perf_counter()-began,'episodes':panels}
  (out/'result.json').write_text(json.dumps(result,indent=2));print('DONE',seed,variant,round(result['survival']*100,2),flush=True)
 print('WORKER COMPLETE',worker,flush=True)
if __name__=='__main__':main()

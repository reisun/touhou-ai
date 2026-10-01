"""Recompare joint18 against adopted independent9+2 baselines."""
import sys,pathlib,json,time,functools,hashlib
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.joint_movement import JointMovementPolicy
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu,distributions
from scripts.continue_dodge_comparison import Progress

OUT=pathlib.Path('artifacts/autumn-joint-adopted-20260929')
def decode_joint(p,u=None):
 p=p.reshape(9,2)
 if u is None:
  index=int(p.argmax());return [index//2,0,index%2,0]
 marginal=p.sum(1);direction=min(int(np.searchsorted(np.cumsum(marginal),u[0])),8)
 if marginal[direction]<=0:raise ValueError('sampled zero mass')
 focus=min(int(np.searchsorted(np.cumsum(p[direction]/marginal[direction]),u[2])),1)
 return [direction,0,focus,0]

def assess(model,cls,n=96,mode='sample'):
 rows=[];model.policy.set_training_mode(False)
 for offset in range(0,n,8):
  envs=[cls() for _ in range(min(8,n-offset))];obs=[e.reset(seed=5000+offset+i)[0] for i,e in enumerate(envs)]
  rngs=[np.random.default_rng(19000+offset+i) for i in range(len(envs))];active=list(range(len(envs)));frozen=None
  counts=np.zeros(len(envs));slow=np.zeros(len(envs));switch=np.zeros(len(envs));last=np.zeros(len(envs))
  if mode.startswith('frozen_'):
   with torch.inference_mode():frozen=distributions(model.policy,{k:np.stack([o[k] for o in obs]) for k in obs[0]})[0].probs.numpy()
   for i,e in enumerate(envs):e.unwrapped.observe=lambda initial=obs[i]:initial
  for step in range(300):
   if frozen is not None:probs=frozen[active]
   else:
    with torch.inference_mode():ds=distributions(model.policy,{k:np.stack([obs[i][k] for i in active]) for k in obs[0]})
    assert len(ds)==1 and ds[0].probs.shape[1]==18
    probs=ds[0].probs.numpy()
   following=[]
   for j,i in enumerate(active):
    a=decode_joint(probs[j],None if mode in ['greedy','frozen_greedy'] else rngs[i].random(4))
    counts[i]+=1;slow[i]+=a[2];switch[i]+=a[2]!=last[i];last[i]=a[2]
    obs[i],_,done,_,info=envs[i].step(a)
    if done:rows.append({'seed':5000+offset+i,'success':info['success'],'frames':info['frames'],'end_xy':envs[i].pos.tolist(),'focus_fraction':slow[i]/counts[i],'switches_per_100_actions':100*switch[i]/counts[i]})
    else:following.append(i)
   active=following
   if not active:break
  if active:raise AssertionError('unfinished evaluation')
 return {'n':n,'survival':float(np.mean([r['success'] for r in rows])),'mean_frames':float(np.mean([r['frames'] for r in rows])),'episodes':rows}

def main():
 rep=sys.argv[1];seeds=[int(x) for x in sys.argv[2:]] or [7,17,27]
 assert rep in ['relative','cnn'];config=json.loads(pathlib.Path('configs/autumn-learning-reference-v2.json').read_text())
 alg={'PPO':PPO,'SeparateClipPPO':SeparateClipPPO}[config['representations'][rep]['algorithm']]
 cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
 baseline=json.loads(pathlib.Path('artifacts/autumn-separate-clip-20260928/holdout.json').read_text())
 baseline={r['seed']:r for r in baseline if r['representation']==rep and r['variant']==('baseline' if rep=='relative' else 'separate')};assert len(baseline)==3
 for seed in seeds:
  out=OUT/rep/str(seed);out.mkdir(parents=True,exist_ok=True)
  if (out/'result.json').exists():continue
  manifest={'representation':rep,'seed':seed,'steps':16384,'config':config,'baseline_source':'artifacts/autumn-separate-clip-20260928/holdout.json','baseline_variant':baseline[seed]['variant'],'physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'change':'joint categorical18 instead of independent direction9+focus2; no risk inputs or auxiliary losses','evaluation':'96 seeds5000..5095; uniforms19000..19095; sample joint via direction marginal and focus conditional using u0/u2; four normal/frozen and stochastic/greedy modes','baseline_reuse':'adopted same-seed independent checkpoints already evaluated with identical environment and RNG seeds'}
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'baseline.json').write_text(json.dumps(baseline[seed],indent=2))
  print('START',rep,seed,flush=True);started=time.perf_counter();torch.set_num_threads(1)
  kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
  if rep=='cnn':kw['features_extractor_class']=NarrowGridFeatures
  m=alg(JointMovementPolicy,cls(),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0,**config['ppo']);tune_cpu(m)
  m.learn(16384,callback=Progress());m.save(out/'joint')
  evaluations={}
  for mode in ['sample','greedy','frozen_sample','frozen_greedy']:
   evaluations[mode]=assess(m,cls,mode=mode);e=evaluations[mode]
   print('EVAL',rep,seed,mode,round(e['survival']*100,2),round(e['mean_frames']/60,3),flush=True)
   (out/'partial.json').write_text(json.dumps(evaluations,indent=2))
  row={'representation':rep,'seed':seed,'evaluations':evaluations,'seconds':time.perf_counter()-started,'training_metrics':{k:float(v) for k,v in m.logger.name_to_value.items() if k.startswith('train/')}}
  (out/'result.json').write_text(json.dumps(row,indent=2));print('COMPLETE',rep,seed,flush=True)
if __name__=='__main__':main()

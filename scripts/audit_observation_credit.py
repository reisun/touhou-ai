"""Read-only checkpoint diagnosis. Counterfactual labels are simulator diagnostics only."""
import sys,pathlib,json,copy,collections,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.fixed_dodge import DIRECTIONS
from touhou_ai.simulation_speed import distributions

FIELDS=['pos','boss','frame','focus','dead','xy','vel','angle','birth','group','alive','n','move_angle','emissions','rng']
def snapshot(e):return {k:copy.deepcopy(getattr(e,k)) for k in FIELDS}
def restore(state,grid=False):
 e=GridAblation() if grid else NumericalAblation(relative=True)
 for k,v in state.items():setattr(e,k,copy.deepcopy(v))
 return e

def branch(state):
 safe2=[];safe16=[]
 for direction in range(9):
  for focus in range(2):
   e=restore(state);e.observe=lambda:{}
   _,_,done,_,_=e.step([direction,0,focus,0]);safe2.append(not e.dead)
   for _ in range(7):
    if done:break
    _,_,done,_,_=e.step([0,0,focus,0])
   safe16.append(not e.dead)
 return np.array(safe2),np.array(safe16)

def linear_prediction(state):
 alive=state['alive'];p=state['xy'][alive];v=state['vel'][alive];origin=state['pos']
 near=np.linalg.norm(p-origin,axis=1)<=np.linalg.norm(v,axis=1)*16+16
 p=p[near];v=v[near];t=np.arange(.25,16.01,.25)
 result=[]
 for direction in range(9):
  for focus in range(2):
   pv=DIRECTIONS[direction]*(2 if focus else 4.5)
   player=np.clip(origin+np.minimum(t[:,None],2)*pv,[-184,32],[184,432])
   bullet=p[None,:,:]+t[:,None,None]*v[None,:,:]
   result.append(not np.any(np.all(np.abs(bullet-player[:,None,:])<=3,axis=2)))
 return np.array(result)

def collect(model):
 selected=[];trajectory=[]
 for ep in range(16):
  e=NumericalAblation(relative=True);o,_=e.reset(seed=2000+ep);rng=np.random.default_rng(17000+ep);history=[]
  for step in range(300):
   tensor,_=model.policy.obs_to_tensor(o)
   with torch.inference_mode():
    pi,vf=model.policy._latents(tensor);ds=model.policy._scheduled(pi,tensor).distribution
    probs=[d.probs[0].numpy() for d in ds];value=float(model.policy.value_net(vf)[0,0])
   u=rng.random(4);a=[min(int(np.searchsorted(np.cumsum(p),u[h])),len(p)-1) for h,p in enumerate(probs)]
   history.append({'scene':snapshot(e),'value':value,'action':a})
   o,reward,done,_,info=e.step(a)
   history[-1]['reward']=reward
   if done:break
  ret=0.;gae=0.;next_v=0.;n=len(history)
  for j in range(n-1,-1,-1):
   h=history[j];ret=h['reward']+.9995*ret
   delta=h['reward']+.9995*next_v-h['value'];gae=delta+.9995*.95*gae;next_v=h['value']
   h.update(episode=ep,step=j,to_end=n-1-j,death=e.dead,mc_return=ret,diagnostic_gae=gae)
  for j,h in enumerate(history):
   trajectory.append({k:v for k,v in h.items() if k!='scene'})
   if (j>=n-8 or j%40==0) and h['scene']['frame']<=584:selected.append(h)
 print('collected',len(trajectory),'transitions',len(selected),'candidate scenes',flush=True)
 rng=np.random.default_rng(47);indices=rng.choice(len(selected),min(128,len(selected)),replace=False)
 return [selected[i] for i in sorted(indices)],trajectory

def stats(rows):
 if not rows:return {'n':0}
 v=np.array([r['value'] for r in rows]);target=np.array([r['mc_return'] for r in rows]);adv=np.array([r['diagnostic_gae'] for r in rows])
 return {'n':len(rows),'mean_value':float(v.mean()),'mean_realized_return':float(target.mean()),'mae_to_realized_return':float(abs(v-target).mean()),'positive_gae_fraction':float((adv>0).mean()),'mean_gae':float(adv.mean())}

def main():
 torch.set_num_threads(1);out=pathlib.Path('artifacts/observation-credit-audit-20260928');out.mkdir(exist_ok=True)
 paths={'numerical-independent':'artifacts/autumn-ablation-20260928/relative-switch-7.zip','numerical-joint':'artifacts/autumn-joint-20260928/numerical/joint-7.zip','cnn-independent':'artifacts/autumn-ablation-20260928/cnn-switch-7.zip','cnn-joint':'artifacts/autumn-joint-20260928/cnn/joint-7.zip'}
 base=PPO.load(paths['numerical-independent'],device='cpu');scenes,traj=collect(base)
 for i,h in enumerate(scenes):
  h['safe2'],h['safe16']=branch(h['scene']);h['linear_safe16']=linear_prediction(h['scene'])
  assert bool(h['safe2'][h['action'][0]*2+h['action'][2]]) == (not (h['death'] and h['to_end']==0)), 'snapshot replay must match observed death'
  if (i+1)%32==0:print('branches',i+1,flush=True)
 metrics={};details=[]
 for name,path in paths.items():
  m=PPO.load(path,device='cpu');allp=[];allv=[]
  for offset in range(0,len(scenes),16):
   observations=[restore(h['scene'],grid=name.startswith('cnn')).observe() for h in scenes[offset:offset+16]]
   batch={k:np.stack([o[k] for o in observations]) for k in observations[0]}
   tensor,_=m.policy.obs_to_tensor(batch)
   with torch.inference_mode():
    ds=distributions(m.policy,batch)
    if len(ds)==1:p=ds[0].probs.numpy()
    else:p=(ds[0].probs[:,:,None]*ds[2].probs[:,None,:]).reshape(-1,18).numpy()
    values=m.policy.predict_values(tensor).flatten().numpy()
   allp.extend(p);allv.extend(values)
  p=np.array(allp);values=np.array(allv);result={}
  for label in ['safe2','safe16']:
   safe=np.stack([h[label] for h in scenes]);mixed=safe.any(1)&(~safe).any(1)
   result[label]={'mixed_scene_count':int(mixed.sum()),'all_actions_fail':int((~safe.any(1)).sum()),'all_actions_survive':int(safe.all(1).sum()),'policy_survival_mass':float((p[mixed]*safe[mixed]).sum(1).mean()) if mixed.any() else None,'uniform_survival_mass':float(safe[mixed].mean()) if mixed.any() else None,'greedy_survival_fraction':float(safe[np.arange(len(safe)),p.argmax(1)][mixed].mean()) if mixed.any() else None}
  result['value_mean_on_same_scenes']=float(values.mean());metrics[name]=result;print(name,json.dumps(result),flush=True)
  details.append({'model':name,'probabilities':p.tolist(),'values':values.tolist()})
 actual=np.stack([h['safe16'] for h in scenes]);pred=np.stack([h['linear_safe16'] for h in scenes])
 episode_errors=np.array([np.mean([r['value']-r['mc_return'] for r in traj if r['episode']==ep]) for ep in range(16)])
 rng=np.random.default_rng(123);boot=episode_errors[rng.integers(0,16,size=(10000,16))].mean(1)
 calibration={'mean_episode_prediction_minus_return':float(episode_errors.mean()),'episode_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'episodes':16}
 report={'episode_calibration':calibration,'collector':'relative independent seed7 policy;16 fresh evaluation episodes2000..2015','scenes':len(scenes),'selection':'last8 preterminal states plus every40 decisions; uniform subset up to128; frame<=584 ensures full16F horizon; not a natural-frequency state sample','horizons':'2F chosen action;16F chosen action for2F then stop14F; simulation counterfactual, not optimal multi-step safety','models':metrics,'critic':{'all':stats(traj),'death_far_over30_decisions':stats([r for r in traj if r['death'] and r['to_end']>30]),'death_last10_decisions':stats([r for r in traj if r['death'] and r['to_end']<10]),'success':stats([r for r in traj if not r['death']])},'constant_velocity_prediction':{'agreement':float((actual==pred).mean()),'predicted_safe_but_hit':int((pred&~actual).sum()),'predicted_hit_but_safe':int((~pred&actual).sum()),'total_action_scenes':int(actual.size)},'credit_weights':{str(n):(.9995*.95)**n for n in [8,30,60,150]},'limitations':'saved policies; diagnostic episode-complete GAE not recorded historical minibatch updates; realized returns include stochastic outcome noise;16F hold reference not proof of globally safe actions; scene sample biased near death; critic calibration only collector on-policy; raw GAE sign is not actual PPO update sign because minibatch advantages are normalized'}
 (out/'report.json').write_text(json.dumps(report,indent=2));(out/'trajectory.json').write_text(json.dumps(traj,indent=2));(out/'policy-details.json').write_text(json.dumps(details))
 serial=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in h.items() if k!='scene'} for h in scenes]
 (out/'scene-labels.json').write_text(json.dumps(serial,indent=2))
 print('REPORT',json.dumps(report),flush=True)
if __name__=='__main__':main()

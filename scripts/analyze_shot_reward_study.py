"""Compare common observations and physical gameplay metrics for both arms."""
import json,pathlib,sys,statistics
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.checkpoint_rng import load_preserving_rng
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.live_action_grid import LiveActionGridContract
root=pathlib.Path(__file__).resolve().parents[1];out=root/'artifacts/shot-reward-study-20260930'
study=json.loads((out/'study.json').read_text());capture=json.loads((root/'artifacts/boss-reward-tanh-20260930/capture.json').read_text())
torch.set_num_threads(1);env=LiveActionGridContract();observations=[]
for row in capture['observations']:
 obs=env.encode(row['raw'],row['rewards'])
 if row['bomb_clock'] is not None:obs['bomb_clock']=np.asarray([((round(row['bomb_clock'][0]*12)+2)%12)/12],np.float32)
 observations.append(obs)
checkpoints=[('common',study['source'])]
performance={}
for name,arm in study['arms'].items():
 scale=arm['status']['shot_reward_study']['scale'];performance[name]=[]
 for e in arm['status']['episodes']:
  checkpoints.append((f'{name}-{e["episode"]}',str(pathlib.Path(arm['record']['Output'])/e['checkpoint'])))
  m=e['shot_study_metrics'];seconds=e['steps']/30
  performance[name].append(dict(episode=e['episode'],game_seconds=seconds,hits=e['hits'],progress=e['max_progress'],
   first_hit_seconds=m['first_hit_decision']/30 if m['first_hit_decision'] is not None else None,
   shot_damage_hp=m['shot_damage_hp'],shot_hp_per_second=m['shot_damage_hp']/seconds,
   shoot_fraction=m['shoot_decisions']/e['steps'],reward_original_scale=e['return']+m['shot_reward']*(1/scale-1),
   terminated=e['terminated'],recovery=e['recovery_update'],optimization=e['optimization']))
diagnostics={}
for label,path in checkpoints:
 model=load_preserving_rng(SeparateClipPPO,pathlib.Path(path),device='cpu');model.policy.set_training_mode(False)
 captured={};handles=[]
 def hook(name):
  def collect(module,args,value):captured.setdefault(name,[]).append(value.detach().numpy().copy())
  return collect
 for name,module in model.policy.named_modules():
  if isinstance(module,torch.nn.Tanh):handles.append(module.register_forward_hook(hook(name)))
 values=[];probabilities=[]
 with torch.no_grad():
  for obs in observations:
   _,value,_,ds=model.policy.forward_with_distribution(model.policy.obs_to_tensor(obs)[0])
   values.append(value.item());probabilities.append(np.concatenate([d.probs[0].numpy() for d in ds]).tolist())
 for h in handles:h.remove()
 layers={}
 for name,parts in captured.items():
  a=np.concatenate(parts)
  layers[name]=dict(saturation=float(np.mean(np.abs(a)>.99)),scene_std_rms=float(np.sqrt(np.mean(a.std(axis=0)**2))),mean_derivative=float(np.mean(1-a*a)))
 diagnostics[label]=dict(checkpoint=path,values=values,value_std=float(np.std(values)),value_range=float(np.ptp(values)),layers=layers,probabilities=probabilities)
 print(label,'critic saturation',layers.get('mlp_extractor.value_net.3'),flush=True)
 del model
summary={}
for name,rows in performance.items():
 seconds=sum(r['game_seconds'] for r in rows)
 summary[name]=dict(games=len(rows),decisions=round(seconds*30),
  mean_game_seconds=statistics.mean(r['game_seconds'] for r in rows),
  mean_first_hit_seconds=statistics.mean(r['first_hit_seconds'] for r in rows if r['first_hit_seconds'] is not None),
  mean_shot_damage_hp=statistics.mean(r['shot_damage_hp'] for r in rows),
  pooled_shot_hp_per_second=sum(r['shot_damage_hp'] for r in rows)/seconds,
  pooled_shoot_fraction=sum(r['shoot_fraction']*r['game_seconds'] for r in rows)/seconds,
  mean_reward_original_scale=statistics.mean(r['reward_original_scale'] for r in rows),
  completed_epochs=[r['optimization']['completed_epochs'] for r in rows])
report=dict(observation_samples=len(observations),performance=performance,summary=summary,diagnostics=diagnostics,
 limitations=['Three training games per arm, not independent frozen-policy evaluation.',
  'Game randomness is not controlled; training decision counts and actual epochs differ.',
  'Previous rewards are policy inputs: scaling changes both live inputs and learning targets.',
  'Diagnostics hold all 20 observation inputs, including previous rewards, identical across checkpoints.',
  'These boss observations are a small historical diagnostic set, not proof of generalization.'])
(out/'analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(performance),flush=True)

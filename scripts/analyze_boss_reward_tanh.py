"""Match critic/actor saturation on captured observations to live reward windows."""
import json,pathlib,sys,statistics
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.checkpoint_rng import load_preserving_rng
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.live_action_grid import LiveActionGridContract
root=pathlib.Path(__file__).resolve().parents[1]/'artifacts/boss-reward-tanh-20260930'
data=json.loads((root/'capture.json').read_text());torch.set_num_threads(1)
segments=[];current=None
for t in data['timeline']:
 if not t['boss']:
  current=None;continue
 if current is None or t['episode']!=current['episode'] or t['time']-current['end']>1:
  current=dict(episode=t['episode'],start=t['time'],end=t['time'],updates=t['updates']);segments.append(current)
 current['end']=t['time']
def summarize(rows):
 duration=len(rows)/30
 return dict(decisions=len(rows),game_seconds=duration,
  damage=sum(r['components'].get('damage',0) for r in rows),
  damage_per_second=sum(r['components'].get('damage',0) for r in rows)/duration,
  all_reward=sum(r['reward'] for r in rows),
  value_mean=statistics.mean(r['value'] for r in rows),value_min=min(r['value'] for r in rows),value_max=max(r['value'] for r in rows))
windows=[];segment_summary=[]
for s in segments:
 rows=sorted([r for r in data['rows'] if r['episode']==s['episode'] and s['start']<=r['time']<=s['end'] and not r['terminal']],key=lambda r:r['time'])
 if not rows:continue
 segment_summary.append(s|summarize(rows))
 for seconds in (30,60):
  n=seconds*30
  for start in range(0,len(rows)-n+1,30):
   part=rows[start:start+n]
   windows.append(dict(episode=s['episode'],start=part[0]['time'],end=part[-1]['time'],**summarize(part)))
results=[];env=LiveActionGridContract()
for checkpoint in sorted(set(o['checkpoint'] for o in data['observations'])):
 model=load_preserving_rng(SeparateClipPPO,pathlib.Path(checkpoint),device='cpu');model.policy.set_training_mode(False)
 captured={};handles=[]
 def hook(name):
  def collect(module,args,out):captured[name]=out.detach().cpu().numpy().copy()
  return collect
 for name,module in model.policy.named_modules():
  if isinstance(module,torch.nn.Tanh):handles.append(module.register_forward_hook(hook(name)))
 with torch.no_grad():
  for row in [o for o in data['observations'] if o['checkpoint']==checkpoint]:
   obs=env.encode(row['raw'],row['rewards'])
   if row['bomb_clock'] is not None:
    tick=(round(row['bomb_clock'][0]*12)+2)%12
    obs['bomb_clock']=np.asarray([tick/12],dtype=np.float32)
   captured.clear();_,value,_=model.policy(model.policy.obs_to_tensor(obs)[0],deterministic=True)
   layers={name:dict(saturation=float(np.mean(np.abs(a)>.99)),mean_derivative=float(np.mean(1-a*a))) for name,a in captured.items()}
   results.append(dict(time=row['time'],episode=row['episode'],updates=row['updates'],frame=row['raw']['stage_frame'],
      power=row['raw']['power_raw']/20,value=float(value.item()),layers=layers,checkpoint=checkpoint))
 for h in handles:h.remove()
 del model
for w in windows:
 matching=[r for r in results if r['episode']==w['episode'] and w['start']<=r['time']<=w['end']]
 w['observation_samples']=len(matching)
 w['layer_mean_saturation']={name:statistics.mean(r['layers'][name]['saturation'] for r in matching) for name in (matching[0]['layers'] if matching else [])}
summary={name:dict(mean_saturation=statistics.mean(r['layers'][name]['saturation'] for r in results),
 max_scene_saturation=max(r['layers'][name]['saturation'] for r in results),
 mean_derivative=statistics.mean(r['layers'][name]['mean_derivative'] for r in results)) for name in (results[0]['layers'] if results else [])}
report=dict(segments=segment_summary,observations=results,layers=summary,windows=windows,
 peak_windows={str(seconds):max((w for w in windows if w['game_seconds']==seconds),key=lambda w:w['damage_per_second'],default=None) for seconds in (30,60)})
(root/'analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(segments=segment_summary,observation_samples=len(results),layers=summary,peak_windows=report['peak_windows'])),flush=True)

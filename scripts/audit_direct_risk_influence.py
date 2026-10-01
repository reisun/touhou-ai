import pathlib,json,functools,numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from scripts.compare_risk_auxiliary import make_probe
from touhou_ai.direct_risk import ACTION_RISK
from touhou_ai.simulation_speed import distributions
out=pathlib.Path('artifacts/autumn-direct-risk-20260928');report=[];torch.set_num_threads(1)
for rep in ['relative','cnn']:
 cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
 observations=make_probe(cls)
 for seed in [7,17,27]:
  path=out/rep/f'direct-{seed}.zip'
  if not path.exists():continue
  m=(PPO if rep=='relative' else SeparateClipPPO).load(path,device='cpu');tv=[];changed=[];move_span=[];focus_span=[]
  for start in range(0,len(observations),16):
   group=observations[start:start+16];batch={k:np.stack([o[k] for o in group]) for k in group[0] if k!='risk_targets'};y=np.stack([o['risk_targets'] for o in group]);batch[ACTION_RISK]=y
   with torch.inference_mode():
    ds=distributions(m.policy,batch);p=(ds[0].probs[:,:,None]*ds[2].probs[:,None,:]).reshape(-1,18).numpy()
    batch[ACTION_RISK]=np.zeros_like(y);ds=distributions(m.policy,batch);q=(ds[0].probs[:,:,None]*ds[2].probs[:,None,:]).reshape(-1,18).numpy()
    logits=y@m.policy.action_net.weight[:,-18:].detach().numpy().T
   tv.extend((np.abs(p-q).sum(1)/2).tolist());changed.extend((p.argmax(1)!=q.argmax(1)).tolist());move_span.extend(np.ptp(logits[:,:9],axis=1).tolist());focus_span.extend(np.ptp(logits[:,11:13],axis=1).tolist())
  report.append({'representation':rep,'seed':seed,'scenes':len(observations),'mean_joint_total_variation':float(np.mean(tv)),'max_joint_total_variation':float(np.max(tv)),'greedy_action_changed_fraction':float(np.mean(changed)),'mean_movement_logit_span':float(np.mean(move_span)),'mean_focus_logit_span':float(np.mean(focus_span))})
(out/'input-influence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

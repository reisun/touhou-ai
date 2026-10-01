"""Descriptive dependence of direction and focus in learned joint policies."""
import json,functools,numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.simulation_speed import distributions
from scripts.compare_risk_auxiliary import make_probe
from scripts.compare_joint_adopted import OUT
report=[];torch.set_num_threads(1)
for rep in ['relative','cnn']:
 cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
 obs=make_probe(cls)
 for seed in [7,17,27]:
  path=OUT/rep/str(seed)/'joint.zip'
  if not path.exists():continue
  m=(PPO if rep=='relative' else SeparateClipPPO).load(path,device='cpu');mi=[];tv=[];spread=[]
  for start in range(0,len(obs),16):
   group=obs[start:start+16];batch={k:np.stack([o[k] for o in group]) for k in group[0] if k!='risk_targets'}
   with torch.inference_mode():p=distributions(m.policy,batch)[0].probs.numpy().reshape(-1,9,2)
   pm=p.sum(2,keepdims=True);pf=p.sum(1,keepdims=True);q=pm*pf
   mi.extend((p*np.log(np.maximum(p,1e-30)/np.maximum(q,1e-30))).sum((1,2)).tolist());tv.extend((abs(p-q).sum((1,2))/2).tolist());spread.extend(np.ptp(p[:,:,1]/pm[:,:,0],axis=1).tolist())
  report.append({'representation':rep,'seed':seed,'scenes':len(obs),'mean_mutual_information_nats':float(np.mean(mi)),'mean_total_variation_from_independent_marginals':float(np.mean(tv)),'mean_conditional_focus_range':float(np.mean(spread))})
(OUT/'joint-dependence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

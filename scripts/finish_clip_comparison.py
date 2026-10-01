import sys,pathlib,json,functools,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from scripts.assess_autumn_holdout import assess_holdout
from scripts.compare_autumn_ablation import main as train_baseline

out=pathlib.Path('artifacts/autumn-separate-clip-20260928')
# Fill the missing independent CNN baseline seeds using the unchanged original runner.
existing_only='--existing-only' in sys.argv
reps=[x for x in sys.argv[1:] if not x.startswith('--')] or ['relative','cnn']
if 'cnn' in reps and not existing_only:
 sys.argv=['baseline','artifacts/autumn-clip-baseline-20260928']
 train_baseline([('cnn','switch',17),('cnn','switch',27)])
rows=[];target=out/'holdout.json'
if target.exists():rows=json.loads(target.read_text())
for rep in reps:
 cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
 for seed in [7,17,27]:
  key=f'{rep}-switch-{seed}'
  for variant in ['baseline','separate']:
   if any(r['key']==key and r['variant']==variant for r in rows):continue
   root=(pathlib.Path('artifacts/autumn-ablation-20260928') if rep=='relative' or seed==7 else pathlib.Path('artifacts/autumn-clip-baseline-20260928')) if variant=='baseline' else out
   if existing_only and not (root/(key+'.zip')).exists():continue
   m=PPO.load(root/(key+'.zip'),device='cpu');torch.set_num_threads(1)
   result={}
   for mode in ['sample','greedy','frozen_sample','frozen_greedy']:
    result[mode]=assess_holdout(m,cls,mode=mode)
    print('EVAL',key,variant,mode,round(result[mode]['survival']*100,2),flush=True)
   row={'key':key,'representation':rep,'seed':seed,'variant':variant,'checkpoint':str(root/(key+'.zip')),'evaluations':result}
   rows.append(row);target.write_text(json.dumps(rows,indent=2))
   print('HOLDOUT',key,variant,{k:(round(v['survival']*100,2),round(v['mean_frames']/60,3)) for k,v in result.items()},flush=True)
print('COMPLETE holdout',flush=True)

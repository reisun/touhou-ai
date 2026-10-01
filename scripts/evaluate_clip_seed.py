import sys,pathlib,json,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import GridAblation
from scripts.assess_autumn_holdout import assess_holdout
seed=int(sys.argv[1]);out=pathlib.Path('artifacts/autumn-separate-clip-20260928')
path=out/f'holdout-cnn-{seed}.json';rows=json.loads(path.read_text()) if path.exists() else []
key=f'cnn-switch-{seed}'
for variant in ['baseline','separate']:
 if any(r['variant']==variant for r in rows):continue
 root=(pathlib.Path('artifacts/autumn-ablation-20260928') if seed==7 else pathlib.Path('artifacts/autumn-clip-baseline-20260928')) if variant=='baseline' else out
 m=PPO.load(root/(key+'.zip'),device='cpu');torch.set_num_threads(1)
 result={}
 for mode in ['sample','greedy','frozen_sample','frozen_greedy']:
  result[mode]=assess_holdout(m,GridAblation,mode=mode)
  print('EVAL',key,variant,mode,round(result[mode]['survival']*100,2),flush=True)
 row={'key':key,'representation':'cnn','seed':seed,'variant':variant,'checkpoint':str(root/(key+'.zip')),'evaluations':result}
 rows.append(row);path.write_text(json.dumps(rows,indent=2))
print('COMPLETE',key,flush=True)

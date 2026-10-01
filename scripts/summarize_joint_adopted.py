import pathlib,json,numpy as np
from scripts.compare_joint_adopted import OUT
report={}
for rep in ['relative','cnn']:
 pairs=[]
 for seed in [7,17,27]:
  out=OUT/rep/str(seed)
  if not (out/'result.json').exists():continue
  b=json.loads((out/'baseline.json').read_text());j=json.loads((out/'result.json').read_text())
  pairs.append({'seed':seed,'independent':{k:{'survival':v['survival'],'seconds':v['mean_frames']/60} for k,v in b['evaluations'].items()},'joint':{k:{'survival':v['survival'],'seconds':v['mean_frames']/60} for k,v in j['evaluations'].items()}})
 if not pairs:continue
 means={side:{mode:{metric:float(np.mean([p[side][mode][metric] for p in pairs])) for metric in ['survival','seconds']} for mode in ['sample','greedy','frozen_sample','frozen_greedy']} for side in ['independent','joint']}
 report[rep]={'n_seeds':len(pairs),'pairs':pairs,'means':means,'improved_seeds':sum(p['joint']['sample']['survival']>p['independent']['sample']['survival'] for p in pairs)}
(OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

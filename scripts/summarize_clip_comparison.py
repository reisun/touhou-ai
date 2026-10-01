import json,pathlib,numpy as np
out=pathlib.Path('artifacts/autumn-separate-clip-20260928')
rows=json.loads((out/'holdout.json').read_text());report={}
for rep in ['relative','cnn']:
 pairs=[]
 for seed in [7,17,27]:
  pair={r['variant']:r for r in rows if r['representation']==rep and r['seed']==seed}
  if len(pair)!=2:continue
  stats={}
  for variant,r in pair.items():
   stats[variant]={mode:{'survival':v['survival'],'seconds':v['mean_frames']/60} for mode,v in r['evaluations'].items()}
  pairs.append({'seed':seed,'results':stats})
 if not pairs:continue
 means={v:{mode:{metric:float(np.mean([p['results'][v][mode][metric] for p in pairs])) for metric in ['survival','seconds']} for mode in ['sample','greedy','frozen_sample','frozen_greedy']} for v in ['baseline','separate']}
 report[rep]={'seed_count':len(pairs),'pairs':pairs,'mean':means,'sample_survival_improved_seeds':sum(p['results']['separate']['sample']['survival']>p['results']['baseline']['sample']['survival'] for p in pairs),'sample_time_improved_seeds':sum(p['results']['separate']['sample']['seconds']>p['results']['baseline']['sample']['seconds'] for p in pairs)}
(out/'holdout-summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

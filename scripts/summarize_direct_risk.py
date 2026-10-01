import json,pathlib,numpy as np
out=pathlib.Path('artifacts/autumn-direct-risk-20260928');report={}
for rep in ['relative','cnn']:
 path=out/rep/'results.json'
 if not path.exists():continue
 rows=json.loads(path.read_text());base={r['seed']:r for r in json.loads((out/rep/'baseline.json').read_text())}
 pairs=[]
 for r in rows:
  b=base[r['seed']]
  pairs.append({'seed':r['seed'],'before':{k:{'survival':v['survival'],'seconds':v['mean_frames']/60} for k,v in b['evaluations'].items()},'after':{k:{'survival':v['survival'],'seconds':v['mean_frames']/60} for k,v in r['evaluations'].items()},'probe_before':r['probe_before'],'probe_after':r['probe_after'],'risk_weight_norm':r['risk_weight_norm']})
 means={side:{mode:{metric:float(np.mean([p[side][mode][metric] for p in pairs])) for metric in ['survival','seconds']} for mode in pairs[0][side]} for side in ['before','after']}
 report[rep]={'seed_count':len(pairs),'pairs':pairs,'means':means,'improved_seed_count':sum(p['after']['sample']['survival']>p['before']['sample']['survival'] for p in pairs),'probe_danger_mass':{mode:float(np.mean([p['probe_after']['modes'][mode]['approx_danger_mass'] for p in pairs])) for mode in ['normal','zero','shuffled']}}
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({rep:{k:v for k,v in r.items() if k!='pairs'} for rep,r in report.items()},indent=2))

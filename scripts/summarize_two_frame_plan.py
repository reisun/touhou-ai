"""Compact paired summaries; episode counts are not independent training runs."""
import json,pathlib
OUT=pathlib.Path('artifacts/two-frame-plan-20261002')
def summarize(es):
 frames=sum(x['frames'] for x in es);pairs=sum(x['pairs'] for x in es)
 return dict(episodes=len(es),wins=sum(x['success'] for x in es),completion=sum(x['success'] for x in es)/len(es),mean_frames=frames/len(es),
 vibration=sum(x['vibration_sum'] for x in es)/frames,reversals_per_second=sum(x['reversals'] for x in es)/(frames/60),
 movement_pixels_per_second=sum(x['distance'] for x in es)/(frames/60),stationary_fraction=sum(x['zero_frames'] for x in es)/frames,
 different_pair_fraction=sum(x['different_pairs'] for x in es)/pairs,one_stop_fraction=sum(x['one_stop_pairs'] for x in es)/pairs,
 opposite_pair_fraction=sum(x['opposite_pairs'] for x in es)/pairs,penalty_per_second=sum(x['penalty'] for x in es)/(frames/60))
rows=[];pooled={}
for split in [0,1]:
 for y in ['330','160']:
  all_es=[]
  for seed in [7,17,27]:
   path=OUT/f'{seed}-{split}/result.json'
   if not path.exists():continue
   data=json.loads(path.read_text());es=data['evaluation'][y];all_es+=es
   rows.append(dict(seed=seed,split=split,start=y,**summarize(es)))
  if all_es:pooled[f'{split}-{y}']=summarize(all_es)
initial={}
for split in [0,1]:
 path=OUT/f'initial-{split}.json'
 if path.exists():
  for y,es in json.loads(path.read_text()).items():initial[f'{split}-{y}']=summarize(es)
report=dict(initial=initial,per_seed=rows,pooled=pooled)
(OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

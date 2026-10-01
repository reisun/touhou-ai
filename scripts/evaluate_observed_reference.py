import sys,pathlib,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.autumn_two_sets import TwoSetsNumerical
from touhou_ai.observed_avoidance import choose
out=pathlib.Path('artifacts/observed-reference-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();rows=[];print(out,flush=True)
for start_seed in [10000,20000]:
 episodes=[];start=time.perf_counter()
 for seed in range(start_seed,start_seed+100):
  e=TwoSetsNumerical();e.reset(seed=seed)
  for step in range(900):
   a=choose(e.pos,e.xy[e.alive],e.vel[e.alive],4.)
   _,_,done,trunc,info=e.step(a)
   if done or trunc:episodes.append({'seed':seed,**info});break
 row={'seed_start':start_seed,'successes':sum(r['success'] for r in episodes),'n':len(episodes),'seconds':time.perf_counter()-start,'episodes':episodes};rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='episodes'}),flush=True)

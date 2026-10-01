import sys,pathlib,json,time,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from touhou_ai.avoidance_ensemble import Ensemble
from touhou_ai.autumn_two_sets import TwoSetsGrid
from scripts.evaluate_autumn_two_sets import assess

def main():
 torch.set_num_threads(1);out=pathlib.Path('artifacts/autumn-ensemble3-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();print(out,flush=True)
 paths=['artifacts/autumn-distill-narrow-cnn-20260928-200456/best.zip','artifacts/autumn-distill-narrow-cnn-20260928-202426/round-4.zip','artifacts/autumn-distill-narrow-cnn-20260928-203041/round-0.zip']
 provider=Ensemble([PPO.load(p,device='cpu') for p in paths]);development={}
 for seed in [2000,2100]:
  development[str(seed)]=assess(None,TwoSetsGrid,'greedy',n=24,seed_start=seed,action_provider=provider);print('DEV',seed,development[str(seed)]['two_sets'],flush=True)
 wins=sum(round(v['two_sets']*24) for v in development.values());(out/'development.json').write_text(json.dumps({'paths':paths,'wins':wins,'development':development},indent=2))
 base=pathlib.Path('artifacts/autumn-ensemble-20260928-203606/development.json')
 for _ in range(120):
  rows=json.loads(base.read_text()) if base.exists() else []
  if len(rows)==3:break
  time.sleep(3)
 else:raise RuntimeError('base development unavailable')
 best=max(r['wins'] for r in rows)
 if wins<=best:
  print('Not selected; equal or lower development result than smaller configuration',wins,best,flush=True);return
 (out/'selection.json').write_text(json.dumps({'paths':paths,'wins':wins,'previous_best_wins':best,'selection':'strictly higher development completion count; no test-bank selection'},indent=2))
 final={}
 for seed in [130000,140000]:
  final[str(seed)]=assess(None,TwoSetsGrid,'greedy',n=100,seed_start=seed,action_provider=provider);(out/'final.json').write_text(json.dumps(final,indent=2));print('FINAL',seed,final[str(seed)]['two_sets'],flush=True)
 print('complete',out,flush=True)
if __name__=='__main__':main()

import sys,pathlib,json,time,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from touhou_ai.avoidance_ensemble import Ensemble
from touhou_ai.autumn_two_sets import TwoSetsGrid
from scripts.evaluate_autumn_two_sets import assess

def main():
 torch.set_num_threads(1);out=pathlib.Path('artifacts/autumn-ensemble-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();print(out,flush=True)
 paths=['artifacts/autumn-distill-narrow-cnn-20260928-200456/best.zip','artifacts/autumn-distill-narrow-cnn-20260928-202426/round-4.zip']
 models=[PPO.load(p,device='cpu') for p in paths];rows=[]
 for name,indices in [('original',[0]),('slow',[1]),('ensemble',[0,1])]:
  provider=Ensemble([models[i] for i in indices]);r={}
  for seed in [2000,2100]:r[str(seed)]=assess(None,TwoSetsGrid,'greedy',n=24,seed_start=seed,action_provider=provider)
  row={'candidate':name,'indices':indices,'development':r,'wins':sum(round(v['two_sets']*24) for v in r.values())};rows.append(row);(out/'development.json').write_text(json.dumps(rows,indent=2));print(name,row['wins'],'/48',flush=True)
 best=max(rows,key=lambda x:x['wins']);provider=Ensemble([models[i] for i in best['indices']]);(out/'selection.json').write_text(json.dumps({'candidate':best['candidate'],'paths':[paths[i] for i in best['indices']],'selection':'highest success count on development 2000..2023 and 2100..2123; ties prefer smaller original model'},indent=2))
 final={}
 for seed in [110000,120000]:
  final[str(seed)]=assess(None,TwoSetsGrid,'greedy',n=100,seed_start=seed,action_provider=provider);(out/'final.json').write_text(json.dumps(final,indent=2));print('FINAL',seed,final[str(seed)]['two_sets'],flush=True)
 print('complete',out,flush=True)
if __name__=='__main__':main()

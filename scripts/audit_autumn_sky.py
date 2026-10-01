import sys,pathlib,json,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from touhou_ai.autumn_sky import AutumnNumerical,AutumnSky
from scripts.train_autumn_sky import assess

def main():
 out=pathlib.Path(sys.argv[1]);torch.set_num_threads(1)
 result=out/'observation-controls.json';rows=json.loads(result.read_text()) if result.exists() else []
 for name,cls in [('numerical',AutumnNumerical),('narrow-cnn',AutumnSky)]:
  if any(r['model']==name for r in rows):continue
  path=out/(name+'-16384.zip')
  if not path.exists():continue
  m=PPO.load(path,device='cpu')
  row={'model':name,**{mode:assess(m,cls,mode=mode) for mode in ['frozen_sample','frozen_greedy']}}
  rows.append(row);print(json.dumps({k:({a:v[a] for a in ['survival','mean_frames']} if isinstance(v,dict) else v) for k,v in row.items()}),flush=True)
 (out/'observation-controls.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()

"""Source-derived Autumn Sky diagnostic; independent of the live game."""
import sys,pathlib,time,json,argparse,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_two_sets import TwoSetsGrid as AutumnSky,TwoSetsNumerical as AutumnNumerical
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.simulation_speed import distributions,tune_cpu
from scripts.continue_dodge_comparison import Progress

from scripts.evaluate_autumn_two_sets import assess

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=16384);args=ap.parse_args()
 out=pathlib.Path('artifacts/autumn-two-trained-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();print(out,flush=True)
 (out/'manifest.json').write_text(json.dumps({'environment':'autumn-two-sets-v1','fresh_initialization':True,'completion_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_two_sets.py').read_bytes()).hexdigest(),'physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'safety_timeout_frames':1800,'completion':'all four volleys cleared without hit','training_seed':7,'evaluation_seeds':list(range(1000,1024)),'rewards':{'death':-60},'steps':args.steps,'fidelity':'source-derived; no live trajectory parity'},indent=2))
 rows=[];baselines={k:assess(None,AutumnNumerical,mode=k) for k in ['stop','random']};(out/'baselines.json').write_text(json.dumps(baselines,indent=2))
 print('baselines',json.dumps({k:{a:v[a] for a in ['n','two_sets','mean_seconds']} for k,v in baselines.items()}),flush=True)
 for name,cls in [('numerical',AutumnNumerical),('narrow-cnn',AutumnSky)]:
  kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
  if name=='narrow-cnn':kw['features_extractor_class']=NarrowGridFeatures
  m=PPO(EvasionPolicy,cls(),seed=7,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0)
  tune_cpu(m);start=time.perf_counter()
  for target in [0,args.steps]:
   if target:m.learn(total_timesteps=target,reset_num_timesteps=False,callback=Progress())
   ts=torch.get_rng_state();ns=np.random.get_state()
   row={'model':name,'steps':m.num_timesteps,'parameters':sum(p.numel() for p in m.policy.parameters()),'sample':assess(m,cls,mode='sample'),'greedy':assess(m,cls,mode='greedy'),'seconds':time.perf_counter()-start}
   if target:
    row['frozen_sample']=assess(m,cls,mode='frozen_sample');row['frozen_greedy']=assess(m,cls,mode='frozen_greedy')
   torch.set_rng_state(ts);np.random.set_state(ns);rows.append(row);m.save(out/(name+'-'+str(target)));(out/'results.json').write_text(json.dumps(rows,indent=2))
   print(json.dumps({k:({a:v[a] for a in ['n','two_sets','mean_seconds']} if k in ['sample','greedy','frozen_sample','frozen_greedy'] else v) for k,v in row.items()}),flush=True)
 print('complete',out,flush=True)
if __name__=='__main__':main()

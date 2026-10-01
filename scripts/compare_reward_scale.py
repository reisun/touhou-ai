import sys,pathlib,json,time,hashlib,functools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
import gymnasium as gym
from touhou_ai.autumn_ablation import SpeedPolicy,GridAblation,NumericalAblation
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu
from scripts.train_autumn_sky import assess
from scripts.continue_dodge_comparison import Progress

class ScaledReward(gym.RewardWrapper):
 def __init__(self,cls):super().__init__(cls())
 def reward(self,reward):return reward/60.
 @property
 def pos(self):return self.env.pos

def main(jobs=None):
 out=pathlib.Path(sys.argv[1]);out.mkdir(exist_ok=True);path=out/'results.json'
 rows=json.loads(path.read_text()) if path.exists() else []
 if jobs is None:
  jobs=[(rep,'switch',7) for rep in ['relative','cnn']]
 (out/'manifest.json').write_text(json.dumps({'jobs':jobs,'steps':16384,'reward':{'death':-1},'environment':'autumn-sky-source-v2','physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'relative':'bullet position minus player position, same scale; player absolute position retained; bullet velocity unchanged','replication':'screen seed7; extend17,27 if promising; reward scale only, all other settings unchanged','evaluation':'24 paired environment seeds1000..1023; sample,greedy,frozen_sample,frozen_greedy'},indent=2))
 for rep,speed,seed in jobs:
  key=f'{rep}-{speed}-{seed}'
  if any(r['key']==key for r in rows):continue
  focus={'switch':None,'fast':0,'slow':1}[speed]
  cls=functools.partial(GridAblation if rep=='cnn' else NumericalAblation,fixed_focus=focus,relative=rep=='relative')
  unscaled_cls=cls
  cls=functools.partial(ScaledReward,unscaled_cls)
  kw={'fixed_focus':focus,'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
  if rep=='cnn':kw['features_extractor_class']=NarrowGridFeatures
  print('START',key,flush=True);start=time.perf_counter()
  m=PPO(SpeedPolicy,cls(),seed=seed,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0)
  tune_cpu(m)
  ts=torch.get_rng_state();ns=np.random.get_state();before=assess(m,cls)
  torch.set_rng_state(ts);np.random.set_state(ns)
  m.learn(total_timesteps=16384,callback=Progress());m.save(out/key)
  evaluations={mode:assess(m,cls,mode=mode) for mode in ['sample','greedy','frozen_sample','frozen_greedy']}
  row={'key':key,'representation':rep,'speed':speed,'seed':seed,'before':before,'evaluations':evaluations,'seconds':time.perf_counter()-start,'training_metrics':{k:float(v) for k,v in m.logger.name_to_value.items() if k.startswith('train/')}}
  rows.append(row);path.write_text(json.dumps(rows,indent=2))
  print('RESULT',key,json.dumps({k:{'survival':v['survival'],'seconds':v['mean_frames']/60} for k,v in evaluations.items()}),flush=True)
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()

import sys,pathlib,time,json,torch,cProfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from tests.fixed_dodge_reference import FixedDodge as Reference
from touhou_ai.fixed_dodge import FixedDodge
import touhou_ai.simulation_speed as speed

def main():
 source='artifacts/side-dodge-20260928-170658/narrow-cnn-32768.zip';rows=[]
 for name,cls,threads,updated in [('reference',Reference,2,2),('optimized',FixedDodge,1,8)]:
  torch.set_num_threads(threads);m=PPO.load(source,env=cls(),device='cpu');orig=m.train;times={}
  def train():
   start=time.perf_counter();torch.set_num_threads(updated);orig();torch.set_num_threads(threads);times['update_seconds']=time.perf_counter()-start
  m.train=train;p=cProfile.Profile();start=time.perf_counter();p.enable();m.learn(512,reset_num_timesteps=False);p.disable();total=time.perf_counter()-start
  rows.append({'case':name,'phase':'learn512','seconds':total,**times});print(rows[-1],flush=True)
 m=PPO.load(source,device='cpu');torch.set_num_threads(1);actor_only=speed.distributions
 for name,cls,batch,legacy in [('reference',Reference,1,True),('optimized_serial',FixedDodge,1,False),('optimized_batch',FixedDodge,16,False)]:
  def reference_dist(policy,obs):
   t,_=policy.obs_to_tensor(obs);return policy.get_distribution(t).distribution
  speed.distributions=reference_dist if legacy else actor_only
  start=time.perf_counter();result=speed.assess_batch(m,cls,n=32,batch_size=batch);rows.append({'case':name,'phase':'evaluate32','seconds':time.perf_counter()-start,'result':result});print(rows[-1],flush=True)
 speed.distributions=actor_only
 # Encoding microbenchmark on the same recorded synthetic scene.
 a,b=Reference(),FixedDodge();a.reset();b.reset()
 for name,e in [('reference',a),('optimized',b)]:
  start=time.perf_counter()
  for _ in range(1000):e.observe()
  rows.append({'case':name,'phase':'encode1000_warm','seconds':time.perf_counter()-start})
 pathlib.Path('artifacts/dodge-speed-comparison.json').write_text(json.dumps(rows,indent=2));print(rows[-2:],flush=True)
if __name__=='__main__':main()

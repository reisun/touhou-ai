import sys,pathlib,json,time,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch,numpy as np
from stable_baselines3 import PPO
from touhou_ai.fixed_dodge import FixedDodge
from scripts.ablate_fixed_dodge import NumericalZeroVelocity

def play(m,cls,det,record=False,override=None):
 e=cls();o,_=e.reset();trace=[]
 for step in range(80):
  with torch.no_grad():
   t,_=m.policy.obs_to_tensor(o);ds=m.policy.get_distribution(t).distribution;p=ds[0].probs[0].numpy();fp=float(ds[2].probs[0,1])
  a,_=m.predict(o,deterministic=det)
  if step==0 and override is not None:a=np.array([override[0],0,override[1],0])
  before=e.pos.tolist();frame=e.frame;o,r,d,_,info=e.step(a)
  trace.append({'frame':frame,'pos':before,'action':a.tolist(),'directions':p.tolist(),'focus_probability':fp,'after':e.pos.tolist()})
  if d:break
 return {'success':info['success'],'frame':e.frame,'position':e.pos.tolist(),'actions':dict(collections.Counter(str(t['action'][0]) for t in trace)),'trace':trace if record else None}

def main():
 torch.set_num_threads(2);root=pathlib.Path('artifacts/dodge-long-20260928-163957');out={}
 for name,cls in [('numerical',NumericalZeroVelocity),('narrow-cnn',FixedDodge)]:
  m=PPO.load(root/(name+'-32768.zip'),device='cpu');det=play(m,cls,True,True);torch.manual_seed(123);np.random.seed(123);samples=[];examples={}
  for i in range(100):
   r=play(m,cls,False,True)
   if str(r['success']) not in examples:examples[str(r['success'])]=r
   samples.append({k:v for k,v in r.items() if k!='trace'})
  cf=[{'first':[d,f],**play(m,cls,True,False,(d,f))} for d in range(9) for f in range(2)]
  out[name]={'deterministic':det,'samples':samples,'examples':examples,'first_action_counterfactuals':cf}
  print(name,'success',sum(x['success'] for x in samples),'greedy',det['frame'],det['position'],det['actions'],'firstaction_success',[x['first'] for x in cf if x['success']],flush=True)
  for t in det['trace'][::5]:print(json.dumps(t),flush=True)
  pathlib.Path('artifacts/dodge-greedy-audit.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()

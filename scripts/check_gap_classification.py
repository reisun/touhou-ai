"""Supervised side decoding with PPO actor architecture; no PPO/live mutation."""
import sys,pathlib,time,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BaseModel
from touhou_ai.side_dodge import SideGrid,SideNumerical
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.evasion_only import EvasionPolicy

def dataset(cls,n,seed):
 rng=np.random.default_rng(seed);obs=[];labels=[]
 for _ in range(n):
  pos=[float(rng.uniform(-150,150)),float(rng.uniform(300,432))];frame=int(rng.integers(0,65));focus=int(rng.integers(2))
  for side in [-1,1]:
   e=cls();e.reset(options={'side':side});e.pos=np.array(pos);e.frame=frame;e.focus=focus
   for b in e.bullets:b['position'][1]+=frame*3
   obs.append(e.observe());labels.append(int(side==1))
 return {k:np.stack([o[k] for o in obs]).astype(np.float16 if k.endswith('_grid') else np.float32) for k in obs[0]},np.array(labels)

def main():
 torch.set_num_threads(4);out=pathlib.Path('artifacts/side-classifier-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();results=[]
 for name,cls in [('numerical',SideNumerical),('narrow-cnn',SideGrid)]:
  torch.manual_seed(7);np.random.seed(7);start=time.perf_counter();kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
  if name=='narrow-cnn':kw['features_extractor_class']=NarrowGridFeatures
  m=PPO(EvasionPolicy,cls(),seed=7,device='cpu',n_steps=8,batch_size=8,policy_kwargs=kw);policy=m.policy;head=nn.Linear(128,2)
  params=list(policy.pi_features_extractor.parameters())+list(policy.mlp_extractor.policy_net.parameters())+list(head.parameters());optimizer=torch.optim.Adam(params,lr=.0001)
  train,labels=dataset(cls,256,31);test,test_labels=dataset(cls,128,47)
  def logits(data,ids):
   tensors={k:torch.as_tensor(v[ids],dtype=torch.float32) for k,v in data.items()}
   features=BaseModel.extract_features(policy,tensors,policy.pi_features_extractor)
   return head(policy.mlp_extractor.forward_actor(features))
  def score(data,y):
   with torch.no_grad():pred=torch.cat([logits(data,np.arange(i,min(i+64,len(y)))).argmax(1) for i in range(0,len(y),64)]).numpy()
   return {'accuracy':float(np.mean(pred==y)),'left':float(np.mean(pred[y==0]==0)),'right':float(np.mean(pred[y==1]==1)),'paired_both_correct':float(np.mean(np.all((pred==y).reshape(-1,2),axis=1)))}
  history=[];rng=np.random.default_rng(7)
  for step in range(501):
   if step in [0,25,100,500]:
    row={'updates':step,'train':score(train,labels),'heldout':score(test,test_labels)};history.append(row);print(name,json.dumps(row),flush=True)
   if step==500:break
   ids=rng.choice(len(labels),64,replace=False);optimizer.zero_grad();loss=nn.functional.cross_entropy(logits(train,ids),torch.tensor(labels[ids]));loss.backward();optimizer.step()
  masked={k:v.copy() for k,v in test.items()}
  if name=='numerical':masked['state'][:,5:]=0
  else:masked['local_grid'][:,1:4]=0;masked['global_grid'][:,:3]=0
  # Matched left/right examples differ only in bullet information.
  assert all(np.array_equal(v[::2],v[1::2]) for v in masked.values())
  control=score(masked,test_labels);assert control['accuracy']==.5
  initial={k:v for k,v in cls().reset()[0].items()};initial_obs=[]
  for side in [-1,1]:initial_obs.append(cls().reset(options={'side':side})[0])
  initial_data={k:np.stack([o[k] for o in initial_obs]) for k in initial_obs[0]}
  result={'model':name,'history':history,'masked_bullets':control,'initial_states':score(initial_data,np.array([0,1])),'seconds':time.perf_counter()-start};results.append(result)
  torch.save({'encoder':policy.pi_features_extractor.state_dict(),'actor_mlp':policy.mlp_extractor.policy_net.state_dict(),'classifier':head.state_dict()},out/(name+'.pt'))
  (out/'results.json').write_text(json.dumps({'train_pairs':256,'heldout_pairs':128,'seed':7,'updates':500,'batch_size':64,'learning_rate':.0001,'results':results,'limits':'supervised classification of synthetic paired snapshots, no control policy training or real-game transfer evidence'},indent=2));print('done',name,result['seconds'],control,flush=True)
 print(out,flush=True)
if __name__=='__main__':main()

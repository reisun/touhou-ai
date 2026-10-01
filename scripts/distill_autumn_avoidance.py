"""Observed-only teacher and dataset aggregation, identical two-set physics."""
import sys,pathlib,json,time,argparse
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_two_sets import TwoSetsNumerical,TwoSetsGrid
from touhou_ai.observed_avoidance import choose
from touhou_ai.simulation_speed import distributions
from scripts.evaluate_autumn_two_sets import assess

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['numerical','numerical-relative','narrow-cnn'],required=True);ap.add_argument('--resume');ap.add_argument('--slow-only',action='store_true');ap.add_argument('--reflect',action='store_true');ap.add_argument('--capacity',type=int,default=4096);ap.add_argument('--rounds',type=int,default=5);ap.add_argument('--train-start',type=int,default=3000);ap.add_argument('--final-start',type=int,default=10000);args=ap.parse_args()
 name=args.model
 from touhou_ai.relative_autumn import RelativeTwoSetsNumerical
 cls={'numerical':TwoSetsNumerical,'numerical-relative':RelativeTwoSetsNumerical,'narrow-cnn':TwoSetsGrid}[name]
 out=pathlib.Path('artifacts/autumn-distill-'+name+'-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();print(out,flush=True)
 torch.set_num_threads(1);torch.manual_seed(71);rng=np.random.default_rng(71)
 if args.resume:m=PPO.load(args.resume,device='cpu')
 elif name=='numerical-relative':
  from touhou_ai.evasion_only import EvasionPolicy
  m=PPO(EvasionPolicy,cls(),seed=7,device='cpu',policy_kwargs={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}},verbose=0)
 else:m=PPO.load(pathlib.Path('artifacts/autumn-two-trained-20260928-192405')/(name+'-16384.zip'),device='cpu')
 e=cls();o,_=e.reset(seed=args.train_start);capacity=args.capacity;count=0;cursor=0;episode=0;rows=[];best=-1
 data={k:np.empty((capacity,*v.shape),dtype=np.float16 if k.endswith('_grid') else np.float32) for k,v in o.items()};labels=np.empty((capacity,4),dtype=np.int64)
 opt=torch.optim.Adam(m.policy.parameters(),lr=.0003)
 if args.resume:
  initial_result=assess(m,cls,'greedy',n=24,seed_start=2000);best=initial_result['two_sets'];m.save(out/'best');rows.append({'round':-1,'development':initial_result})
 schedule=([1.,.3]+[.1]*max(0,args.rounds-2)) if args.resume else ([1.,.7,.3]+[0.]*max(0,args.rounds-3))
 for round,teacher_mix in enumerate(schedule[:args.rounds]):
  torch.set_num_threads(1);start=time.perf_counter()
  for step in range(1536):
   label=choose(e.pos,e.xy[e.alive],e.vel[e.alive],4.,allow_fast=not args.slow_only)
   for k in data:data[k][cursor]=o[k]
   labels[cursor]=label;cursor=(cursor+1)%capacity;count=min(count+1,capacity)
   if rng.random()<teacher_mix:a=label
   else:
    with torch.inference_mode():a=np.array([int(d.probs.argmax()) for d in distributions(m.policy,o)])
   o,_,done,trunc,_=e.step(a)
   if done or trunc:episode+=1;o,_=e.reset(seed=args.train_start+episode)
  moving=np.flatnonzero(labels[:count,0]!=0);stopping=np.flatnonzero(labels[:count,0]==0)
  torch.set_num_threads(8);losses=[]
  for update in range(400):
   # Balanced moving/holding samples prevent a trivial always-stop classifier.
   if len(moving) and len(stopping):ids=np.concatenate((rng.choice(moving,32),rng.choice(stopping,32)))
   else:ids=rng.integers(count,size=64)
   obs={k:v[ids].astype(np.float32) for k,v in data.items()};target=labels[ids].copy()
   if args.reflect:
    from touhou_ai.avoidance_augmentation import reflect_batch
    reflect_batch(obs,target,np.flatnonzero(rng.random(len(ids))<.5))
   a=torch.as_tensor(target);ds=distributions(m.policy,obs)
   loss=-(ds[0].log_prob(a[:,0])+ds[2].log_prob(a[:,2])).mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.policy.parameters(),.5);opt.step();losses.append(float(loss.detach()))
  torch.set_num_threads(1)
  result=assess(m,cls,'greedy',n=24,seed_start=2000)
  row={'round':round,'teacher_mix':teacher_mix,'dataset_count':count,'moving_samples':len(moving),'loss':float(np.mean(losses[-50:])),'development':result,'seconds':time.perf_counter()-start};rows.append(row)
  m.save(out/('round-'+str(round)));(out/'results.json').write_text(json.dumps(rows,indent=2))
  if result['two_sets']>best:best=result['two_sets'];m.save(out/'best')
  print(json.dumps({'round':round,'successes':round_int(result['two_sets']*24),'loss':row['loss'],'seconds':row['seconds']}),flush=True)
 del data
 m=PPO.load(out/'best',device='cpu')
 # Two sealed pattern banks, opened once after development model selection.
 final={str(seed):assess(m,cls,'greedy',n=100,seed_start=seed) for seed in [args.final_start,args.final_start+10000]}
 (out/'final.json').write_text(json.dumps(final,indent=2));print('FINAL',json.dumps({k:v['two_sets'] for k,v in final.items()}),flush=True)
 (out/'protocol.json').write_text(json.dumps({'teacher':'observed pos/velocity only, sigma4','training_seeds':str(args.train_start)+'+episode','resume':args.resume,'development_seeds':'2000..2023','final_seeds':[args.final_start,args.final_start+10000],'requirement':'.95 on both final banks','horizontal_reflection':args.reflect,'slow_only_teacher':args.slow_only,'capacity':args.capacity,'algorithm':'supervised imitation plus dataset aggregation; not PPO-only','physics':'unchanged two-set source-derived environment'},indent=2))
 print('complete',out,flush=True)
def round_int(x):return int(np.rint(x))
if __name__=='__main__':main()

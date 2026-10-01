"""CPU-only rapid diagnostic. Does not modify or pause the live learner."""
import sys,pathlib,json,time,argparse
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.fixed_dodge import FixedDodge
from touhou_ai.numerical_dodge import NumericalDodge
from touhou_ai.dual_grid import DualGridFeatures,GridRolloutBuffer
from touhou_ai.evasion_only import EvasionPolicy

def evaluate_legacy(model,n=50,deterministic=False,env_class=FixedDodge):
    env=env_class();wins=0
    for _ in range(n):
        obs,_=env.reset();done=False
        while not done:
            a,_=model.predict(obs,deterministic=deterministic);obs,_,done,_,info=env.step(a)
        wins+=info['success']
    return wins/n

def evaluate(model,n=50,deterministic=False,env_class=FixedDodge):
    from touhou_ai.simulation_speed import assess_batch
    return assess_batch(model,env_class,n=n,det=deterministic)['survival']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=16384);parser.add_argument('--numerical',action='store_true');args=parser.parse_args()
    env_class=NumericalDodge if args.numerical else FixedDodge
    policy_kwargs={'share_features_extractor':False,'net_arch':{'pi':[64,64],'vf':[64,64]}} if args.numerical else {'features_extractor_class':DualGridFeatures,'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
    torch.set_num_threads(2);out=pathlib.Path(('artifacts/numerical-dodge-' if args.numerical else 'artifacts/fixed-dodge-')+time.strftime('%Y%m%d-%H%M%S'));out.mkdir()
    model=PPO(EvasionPolicy,env_class(),seed=7,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=policy_kwargs,verbose=0)
    from touhou_ai.simulation_speed import tune_cpu
    tune_cpu(model)
    results=[];start=time.perf_counter()
    for target in [0,args.steps//2,args.steps]:
        if target:model.learn(total_timesteps=target-model.num_timesteps,reset_num_timesteps=False)
        # Preserve training RNG across assessment.
        ts=torch.get_rng_state();ns=np.random.get_state();torch.manual_seed(123);np.random.seed(123)
        row={'evaluation_protocol':'numpy-per-episode-123-batch16-v1','steps':model.num_timesteps,'success_rate':evaluate(model,env_class=env_class),'deterministic_success':evaluate(model,1,True,env_class),'elapsed_seconds':time.perf_counter()-start}
        torch.set_rng_state(ts);np.random.set_state(ns);results.append(row);print(json.dumps(row),flush=True)
        (out/'result.json').write_text(json.dumps({'task':'fixed wall with gap x32..64; terminal survival160F; death -60 only','seed':7,'numerical':args.numerical,'parameters':sum(p.numel() for p in model.policy.parameters()),'results':results,'limitations':'synthetic physics; fixed scene; fixed512-step rollout unlike live game-over updates; not real-game validation'},indent=2));model.save(out/'latest.zip')
    print(out,flush=True)
if __name__=='__main__':main()

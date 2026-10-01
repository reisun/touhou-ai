import sys,pathlib,time,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch,numpy as np
from stable_baselines3 import PPO
from touhou_ai.corridor_dodge import CorridorGrid as SideGrid,CorridorNumerical as SideNumerical
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.evasion_only import EvasionPolicy
from scripts.continue_dodge_comparison import Progress

def assess_legacy(m,cls,swapped=False,n=50,det=False):
    result={};torch.manual_seed(123)
    for side in [-1,1]:
        e=cls();wins=0;xs=[]
        for _ in range(n):
            o,_=e.reset(options={'side':side})
            for step in range(80):
                a,_=m.predict(e.opposite_observation() if swapped else o,deterministic=det)
                o,r,d,_,info=e.step(a)
                if step==9:xs.append(float(e.pos[0]))
                if d:wins+=info['success'];break
        result[str(side)]={'survival':wins/n,'x_after20f':float(np.mean(xs))}
    return result

def assess(m,cls,swapped=False,n=50,det=False):
    from touhou_ai.simulation_speed import assess_batch
    return {str(side):assess_batch(m,cls,n=n,side=side,swapped=swapped,det=det) for side in [-1,1]}

def main():
    torch.set_num_threads(2);out=pathlib.Path('artifacts/corridor-dodge-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();rows=[]
    for name,cls in [('numerical',SideNumerical),('narrow-cnn',SideGrid)]:
        kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
        if name=='narrow-cnn':kw['features_extractor_class']=NarrowGridFeatures
        m=PPO(EvasionPolicy,cls(),seed=7,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0)
        from touhou_ai.simulation_speed import tune_cpu
        tune_cpu(m);start=time.perf_counter()
        for target in [0,8192,16384]:
            if target:m.learn(total_timesteps=target-m.num_timesteps,reset_num_timesteps=False,callback=Progress())
            ts=torch.get_rng_state();ns=np.random.get_state();row={'evaluation_protocol':'numpy-per-episode-123-batch16-v1','model':name,'steps':m.num_timesteps,'normal':assess(m,cls),'greedy':assess(m,cls,n=1,det=True)}
            if target==16384:row['opposite_bullets']=assess(m,cls,True)
            torch.set_rng_state(ts);np.random.set_state(ns);row['seconds']=time.perf_counter()-start;rows.append(row);m.save(out/(name+'-'+str(target)));(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    print(out,flush=True)
if __name__=='__main__':main()

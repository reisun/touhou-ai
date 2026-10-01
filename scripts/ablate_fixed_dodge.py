"""Matched head-size x observation-pipeline ablation; synthetic only."""
import pathlib,sys,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.fixed_dodge import FixedDodge
from touhou_ai.numerical_dodge import NumericalDodge
from touhou_ai.dual_grid import DualGridFeatures,GridRolloutBuffer
from touhou_ai.evasion_only import EvasionPolicy
from scripts.train_fixed_dodge import evaluate
class NumericalZeroVelocity(NumericalDodge):
    def observe(self):
        obs=super().observe();obs['state'][2:4]=0
        return obs

def main():
    torch.set_num_threads(2);out=pathlib.Path('artifacts/dodge-ablation-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();results=[]
    for kind,head in [('numerical','large'),('numerical','small'),('grid','small'),('grid','large')]:
        env=NumericalZeroVelocity if kind=='numerical' else FixedDodge
        net=[64,64] if head=='small' else [256,128]
        kw={'share_features_extractor':False,'net_arch':{'pi':net,'vf':net}}
        if kind=='grid':kw['features_extractor_class']=DualGridFeatures
        start=time.perf_counter()
        if kind=='grid' and head=='large':
            m=PPO.load('artifacts/fixed-dodge-20260928-160900/latest.zip',device='cpu');baseline=None
        else:
            m=PPO(EvasionPolicy,env(),seed=7,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0)
            ts=torch.get_rng_state();ns=np.random.get_state();torch.manual_seed(123);np.random.seed(123);baseline=evaluate(m,50,False,env);torch.set_rng_state(ts);np.random.set_state(ns)
            from touhou_ai.simulation_speed import tune_cpu
            tune_cpu(m)
            m.learn(total_timesteps=8192);m.save(out/(kind+'-'+head))
        torch.manual_seed(123);np.random.seed(123)
        row={'evaluation_protocol':'numpy-per-episode-123-batch16-v1','input':kind,'head':head,'steps':m.num_timesteps,'parameters':sum(p.numel() for p in m.policy.parameters()),'baseline50':baseline,'success100':evaluate(m,100,False,env),'deterministic':evaluate(m,1,True,env),'seconds':time.perf_counter()-start}
        results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
    print(out,flush=True)
if __name__=='__main__':main()

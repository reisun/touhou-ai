"""Matched longer training from existing8192-step checkpoints; no live access."""
import sys,pathlib,time,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from touhou_ai.fixed_dodge import FixedDodge
from scripts.ablate_fixed_dodge import NumericalZeroVelocity
from scripts.train_fixed_dodge import evaluate
class Progress(BaseCallback):
    def _on_step(self):
        if self.num_timesteps%4096==0:print('training',self.num_timesteps,flush=True)
        return True

def main():
    torch.set_num_threads(2);out=pathlib.Path('artifacts/dodge-long-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();rows=[]
    sources=[('numerical',NumericalZeroVelocity,'artifacts/dodge-ablation-20260928-162415/numerical-large.zip'),('narrow-cnn',FixedDodge,'artifacts/narrow-grid-20260928-163325/grid-narrow-large.zip')]
    for name,env,source in sources:
        m=PPO.load(source,env=env(),device='cpu');
        from touhou_ai.simulation_speed import tune_cpu
        tune_cpu(m);torch.manual_seed(19);np.random.seed(19);start=time.perf_counter()
        for target in [8192,16384,32768]:
            if target>m.num_timesteps:m.learn(total_timesteps=target-m.num_timesteps,reset_num_timesteps=False,callback=Progress())
            ts=torch.get_rng_state();ns=np.random.get_state();torch.manual_seed(123);np.random.seed(123)
            row={'evaluation_protocol':'numpy-per-episode-123-batch16-v1','model':name,'steps':m.num_timesteps,'success100':evaluate(m,100,False,env),'deterministic_success':evaluate(m,1,True,env),'seconds':time.perf_counter()-start}
            torch.set_rng_state(ts);np.random.set_state(ns);rows.append(row);m.save(out/(name+'-'+str(target)))
            (out/'results.json').write_text(json.dumps({'continuation_rng':19,'evaluation_rng':123,'sources':[s[2] for s in sources],'results':rows,'note':'Same settings; resumes optimizer, RNG restarted19; no claim of independent seeds or generalization'},indent=2));print(json.dumps(row),flush=True)
    print(out,flush=True)
if __name__=='__main__':main()

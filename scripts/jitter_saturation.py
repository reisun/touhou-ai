import json,numpy as np,torch
from scripts.compare_critic_tanh import Record

def fixed_observations():
    from scripts.live_jitter_trial import Env
    rows=[]
    for seed in (7100,7101,7102,7103):
        e=Env();o,_=e.reset(seed=seed)
        for t in range(120):
            if t%30==0:rows.append(o)
            o,_,d,_,_=e.step([0,0,0,0])
            if d:break
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]}

def measure(policy,obs):
    captured={};hooks=[]
    for name,module in policy.mlp_extractor.named_modules():
        if isinstance(module,torch.nn.Tanh):
            def hook(m,a,y,name=name):captured[name]=float((y.abs()>.99).float().mean())
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():policy(policy.obs_to_tensor(obs)[0],deterministic=True)
    finally:
        for h in hooks:h.remove()
    return captured

class SaturationRecord(Record):
    def __init__(self,path):
        super().__init__(path);self.obs=fixed_observations();self.bad=0;self.stop=False
    def _on_training_start(self):self.base=measure(self.model.policy,self.obs)
    def _on_step(self):return not self.stop
    def _on_rollout_end(self):
        super()._on_rollout_end();sat=measure(self.model.policy,self.obs)
        self.rows[-1]['tanh_saturation']=sat
        self.rows[-1]['baseline_saturation']=self.base
        self.bad=self.bad+1 if any(v>=.95 and v>=self.base[k]+.05 for k,v in sat.items()) else 0
        if self.bad>=3:
            self.stop=True
            (self.path.parent.parent/'saturation-stop.json').write_text(json.dumps(dict(path=str(self.path),steps=self.num_timesteps,baseline=self.base,current=sat)))
        self.path.write_text(json.dumps(self.rows,indent=2))

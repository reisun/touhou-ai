"""Independent input sensitivity study; no production or other trial edits."""
import sys,pathlib,json,functools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.scaled_candidate import ScaledCandidateEnv
from touhou_ai.live_action_grid import LiveActionGridContract
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.simulation_speed import distributions

OUT=pathlib.Path('artifacts/input-structure-jitter-20261001')
SOURCE=pathlib.Path('artifacts/live-learning-20261001-190603-3fbec1/real-episode-14.zip')
class Env(ScaledCandidateEnv):
    def __init__(self):
        self.previous={};super().__init__('action_grid',-1)
        self.encoder=LiveActionGridContract();self.observation_space=self.encoder.observation_space
    def reset(self,**kwargs):self.previous={};return super().reset(**kwargs)
    def observe(self):
        if not isinstance(self.encoder,LiveActionGridContract):return super().observe()
        o=self.encoder.encode(self.raw_observation(),self.previous);o['bomb_clock']=np.array([(self.frame%12)/12],np.float32);return o
    def step(self,a):
        _,r,d,tr,info=super().step(a);self.previous={'hit':r};return self.observe(),r,d,tr,info

VARIANTS=('base','freeze_focus','freeze_clock','freeze_player','freeze_local','freeze_global','freeze_risk','freeze_reward','smooth_focus','smooth_local','smooth_global')
def changed(obs,initial,ema,variant):
    o={k:v.copy() for k,v in obs.items()}
    keys={'focus':'player','clock':'bomb_clock','player':'player','local':'local_grid','global':'global_grid','risk':'action_grid','reward':'previous_rewards'}
    if variant=='base':return o
    kind,part=variant.split('_');key=keys[part];ref=initial if kind=='freeze' else ema
    if part=='focus':o[key][8]=ref[key][8]
    else:o[key]=ref[key].copy()
    return o

def run():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    m=SeparateClipPPO.load(SOURCE,device='cpu');m.policy.set_training_mode(False)
    records=[]
    for seed in range(5000,5016):
        e=Env();o,_=e.reset(seed=seed);e.pos[1]=160 if seed<5008 else 400;o=e.observe();initial={k:v.copy() for k,v in o.items()};ema={k:v.copy() for k,v in o.items()}
        assert m.observation_space==e.observation_space
        rng=np.random.default_rng(19000+seed-5000);trace=[]
        for t in range(300):
            ema={k:.5*ema[k]+.5*v for k,v in o.items()}
            inputs=[changed(o,initial,ema,v) for v in VARIANTS]
            batch={k:np.stack([x[k] for x in inputs]) for k in o}
            with torch.inference_mode():ps=[d.probs.cpu().numpy() for d in distributions(m.policy,batch)]
            u=rng.random(4);a=[min(int(np.searchsorted(np.cumsum(p[0]),u[h])),p.shape[1]-1) for h,p in enumerate(ps)];a[1]=a[3]=0
            trace.append(dict(probabilities=ps[0].tolist(),focus=float(o['player'][8]),clock=float(o['bomb_clock'][0]),action=a))
            o,_,done,_,info=e.step(a)
            if done:break
        records.append(dict(seed=seed,start_y=160 if seed<5008 else 400,success=bool(info['success']),frames=info['frames'],trace=trace));print(seed,len(trace),flush=True)
    summary={}
    for j,v in enumerate(VARIANTS):
        switches=total=0;tvs=[];disagree=0;focus_change_switch=0
        for r in records:
            p=np.array([x['probabilities'][j] for x in r['trace']]);base=np.array([x['probabilities'][0] for x in r['trace']]);b=p.argmax(1);sw=b[1:]!=b[:-1]
            switches+=int(sw.sum());total+=len(sw);tvs.extend((np.abs(p[1:]-p[:-1]).sum(1)/2).tolist());disagree+=int((b!=base.argmax(1)).sum())
            f=np.array([x['focus'] for x in r['trace']]);focus_change_switch+=int((sw&(f[1:]!=f[:-1])).sum())
        summary[v]=dict(switches=switches,transitions=total,switch_fraction=switches/total,mean_tv=float(np.mean(tvs)),changed_argmax=disagree,focus_change_switch=focus_change_switch)
    (OUT/'probe.json').write_text(json.dumps(dict(checkpoint=str(SOURCE.resolve()),variants=VARIANTS,records=records,summary=summary)))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

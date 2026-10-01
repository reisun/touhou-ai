"""Closed-loop input/feature smoothing on one pinned real checkpoint."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.input_jitter_probe import Env,OUT,SOURCE
from scripts.audit_motion_jitter import metrics
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.simulation_speed import distributions
import numpy as np,torch

def run(mode):
    torch.set_num_threads(1);m=SeparateClipPPO.load(SOURCE,device='cpu');m.policy.set_training_mode(False);records=[]
    handles=[];active=[];feature_state={}
    if mode=='blur_local':
        def blur(mod,args):return (torch.nn.functional.avg_pool2d(torch.nn.functional.pad(args[0],(1,1,1,1),mode='replicate'),3,stride=1),)
        for idx in (2,4,6):handles.append(m.policy.pi_features_extractor.local[idx].register_forward_pre_hook(blur))
    if mode=='feature_ema':
        def smooth(mod,args,out):
            result=[]
            for j,i in enumerate(active):
                x=out[j] if i not in feature_state else .5*feature_state[i]+.5*out[j]
                feature_state[i]=x.detach().clone();result.append(x)
            return torch.stack(result)
        handles.append(m.policy.pi_features_extractor.local.register_forward_hook(smooth))
    for group in range(4):
        envs=[Env() for _ in range(8)];observations=[];feature_state.clear()
        for i,e in enumerate(envs):
            e.reset(seed=6000+group*8+i);e.pos[1]=160 if group<2 else 400;observations.append(e.observe())
        ema=[o['local_grid'].copy() for o in observations];rng=[np.random.default_rng(23000+group*8+i) for i in range(8)];trace=[[] for _ in envs];active=list(range(8))
        for t in range(300):
            batch={k:np.stack([observations[i][k] for i in active]) for k in observations[0]}
            if mode=='input_ema':
                for j,i in enumerate(active):ema[i]=.5*ema[i]+.5*observations[i]['local_grid'];batch['local_grid'][j]=ema[i]
            with torch.inference_mode():ps=[d.probs.cpu().numpy() for d in distributions(m.policy,batch)]
            follow=[]
            for j,i in enumerate(active):
                u=rng[i].random(4);a=[min(int(np.searchsorted(np.cumsum(p[j]),u[h])),p.shape[1]-1) for h,p in enumerate(ps)];a[1]=a[3]=0
                old=envs[i].pos.copy();observations[i],_,done,_,info=envs[i].step(a);p=np.sort(ps[0][j])
                trace[i].append(dict(best=int(ps[0][j].argmax()),margin=float(p[-1]-p[-2]),action=a,delta=(envs[i].pos-old).tolist()))
                if done:records.append(dict(seed=6000+group*8+i,start_y=160 if group<2 else 400,success=bool(info['success']),frames=info['frames'],metrics=metrics(trace[i])))
                else:follow.append(i)
            active=follow
            if not active:break
        assert not active
        print(mode,(group+1)*8,flush=True)
    for h in handles:h.remove()
    (OUT/f'trial-{mode}.json').write_text(json.dumps(records,indent=2))
    print(mode,'wins',sum(r['success'] for r in records),flush=True)
if __name__=='__main__':run(sys.argv[1])

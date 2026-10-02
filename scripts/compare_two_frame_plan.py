"""Paired held/split controls from a frozen real checkpoint; offline only."""
import pathlib,sys,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.two_frame_plan_sim import PlanEnv,PlanPolicy
from touhou_ai.spatial_input_candidates import CandidateFeatures
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu
from scripts.compare_critic_tanh import Record
OUT=pathlib.Path('artifacts/two-frame-plan-20261002')

def build(seed,split):
    manifest=json.loads((OUT/'manifest.json').read_text())
    old=SeparateClipPPO.load(manifest['checkpoint'],device='cpu')
    m=SeparateClipPPO(PlanPolicy,PlanEnv(split=split),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,
      policy_kwargs=dict(split=split,share_features_extractor=False,net_arch=dict(pi=[256,128],vf=[256,128]),features_extractor_class=CandidateFeatures),verbose=0,**manifest['ppo'])
    source=old.policy.state_dict();state=m.policy.state_dict()
    for k,v in state.items():
        if k.startswith('action_net.'):continue
        assert k in source and source[k].shape==v.shape,k
        v.copy_(source[k])
    with torch.no_grad():
        for attr in ('weight','bias'):
            dst=getattr(m.policy.action_net,attr);src=getattr(old.policy.action_net,attr)
            dst[:9].copy_(src[:9]);dst[9:90].copy_(src[:9].repeat((9,1) if attr=='weight' else (9,)))
            dst[90:].copy_(src[11:13])
    obs=m.env.envs[0].reset(seed=99)[0]
    tensor=m.policy.obs_to_tensor(obs)[0]
    with torch.inference_mode():
        dist=m.policy.get_distribution(tensor);old_dist=old.policy.get_distribution(tensor)
        assert torch.allclose(dist.first.probs,old_dist.distribution[0].probs,atol=1e-6)
        assert torch.allclose(dist.focus.probs,old_dist.distribution[2].probs,atol=1e-6)
        assert torch.allclose(m.policy.predict_values(tensor),old.policy.predict_values(tensor),atol=1e-6)
    return m

def evaluate(m,split,n=32,start_y=330):
    result=[];m.policy.set_training_mode(False)
    for offset in range(0,n,8):
        envs=[PlanEnv(start_y,split) for _ in range(min(8,n-offset))]
        obs=[e.reset(seed=5000+offset+i)[0] for i,e in enumerate(envs)]
        rng=[np.random.default_rng(19000+offset+i) for i in range(len(envs))];active=list(range(len(envs)))
        traces=[[] for _ in envs]
        for _ in range(300):
            batch={k:np.stack([obs[i][k] for i in active]) for k in obs[0]}
            with torch.inference_mode():
                dist=m.policy.get_distribution(m.policy.obs_to_tensor(batch)[0]);p=dist.first.probs.numpy();q=dist.second.probs.numpy();f=dist.focus.probs.numpy()
            next_active=[]
            for j,i in enumerate(active):
                u=rng[i].random(3)
                def draw(ps,r):return min(len(ps)-1,int(np.searchsorted(np.cumsum(ps),r)))
                a=draw(p[j],u[0]);b=draw(q[j,a],u[1]) if split else a;focus=draw(f[j],u[2])
                obs[i],reward,done,_,info=envs[i].step([a,b,focus]);traces[i].append(dict(action=[a,b,focus],**info))
                if done:
                    deltas=np.array([v for t in traces[i] for v in t['deltas']]);norm=np.linalg.norm(deltas,axis=1)
                    reversals=(np.sum(deltas[1:]*deltas[:-1],axis=1)<-.5*norm[1:]*norm[:-1])&(norm[1:]>.01)&(norm[:-1]>.01)
                    actions=np.array([t['action'] for t in traces[i]])
                    result.append(dict(seed=5000+offset+i,success=bool(info['success']),frames=info['frames'],
                      penalty=sum(t['penalty'] for t in traces[i]),vibration_sum=sum(sum(t['vibration']) for t in traces[i]),
                      distance=float(norm.sum()),zero_frames=int(sum(norm<.01)),reversals=int(sum(reversals)),
                      pairs=len(actions),different_pairs=int(sum(actions[:,0]!=actions[:,1])),
                      one_stop_pairs=int(sum((actions[:,0]==0)^(actions[:,1]==0))),
                      opposite_pairs=int(sum((actions[:,0]>0)&(actions[:,1]>0)&((actions[:,0]-actions[:,1])%8==4))),
                      trace=traces[i]))
                else:next_active.append(i)
            active=next_active
            if not active:break
        assert not active
    return result

if __name__=='__main__':
    torch.set_num_threads(1);seed=int(sys.argv[1]);split=bool(int(sys.argv[2]));dest=OUT/f'{seed}-{int(split)}';dest.mkdir(exist_ok=True)
    m=build(seed,split)
    if '--initial' in sys.argv:
        out={str(y):evaluate(m,split,start_y=y) for y in [330,160]};(OUT/f'initial-{int(split)}.json').write_text(json.dumps(out));sys.exit()
    tune_cpu(m,2);start=time.monotonic()
    if (dest/'model.zip').exists():m=SeparateClipPPO.load(dest/'model.zip',device='cpu')
    else:
        m.learn(8192,callback=Record(dest/'learning.json'));m.save(dest/'model.zip')
    reloaded=SeparateClipPPO.load(dest/'model.zip',device='cpu')
    assert all(torch.equal(v,reloaded.policy.state_dict()[k]) and torch.isfinite(v).all() for k,v in m.policy.state_dict().items())
    results={str(y):evaluate(reloaded,split,start_y=y) for y in [330,160]}
    (dest/'result.json').write_text(json.dumps(dict(seed=seed,split=split,steps=8192,seconds=time.monotonic()-start,evaluation=results)))
    print('complete',seed,split,flush=True)

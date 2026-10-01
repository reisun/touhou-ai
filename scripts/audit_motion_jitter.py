"""Read-only v4 policy trajectories: argmax changes and physical reversals."""
import pathlib,sys,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.scaled_candidate import ScaledCandidateEnv
from touhou_ai.simulation_speed import distributions

def metrics(trace):
    v=np.array([x['delta'] for x in trace]); norms=np.linalg.norm(v,axis=1)
    reversals=(np.sum(v[1:]*v[:-1],axis=1)<-.5*norms[1:]*norms[:-1]) & (norms[1:]>.01)&(norms[:-1]>.01)
    result={}
    for w in (3,6,12):
        valid=flag=0
        for end in range(w,len(v)+1):
            chunk=v[end-w:end];length=np.linalg.norm(chunk,axis=1).sum()
            if length<1:continue
            valid+=1;cancel=1-np.linalg.norm(chunk.sum(axis=0))/length
            flag+=int(cancel>=.75 and reversals[end-w:end-1].sum()>=2)
        result[str(w*2)]=dict(valid=valid,flags=flag,fraction=flag/valid if valid else 0)
    return dict(argmax_switches=sum(a['best']!=b['best'] for a,b in zip(trace,trace[1:])),
        action_switches=sum(a['action'][0]!=b['action'][0] for a,b in zip(trace,trace[1:])),
        non_argmax=sum(x['action'][0]!=x['best'] for x in trace),reversals=int(reversals.sum()),
        mean_margin=float(np.mean([x['margin'] for x in trace])),windows=result)

def run(seed):
    torch.set_num_threads(1)
    model=SeparateClipPPO.load(f'artifacts/critic-normalization-20260930/control/{seed}/model.zip',device='cpu');model.policy.set_training_mode(False)
    out=pathlib.Path('artifacts/motion-jitter-20261001');out.mkdir(exist_ok=True);results=[]
    for mode in ('sample','greedy'):
        for offset in range(0,96,8):
            envs=[ScaledCandidateEnv('action_grid',death_reward=-1) for _ in range(8)]
            obs=[e.reset(seed=5000+offset+i)[0] for i,e in enumerate(envs)]
            rng=[np.random.default_rng(19000+offset+i) for i in range(8)];traces=[[] for _ in envs];active=list(range(8))
            for t in range(300):
                batch={k:np.stack([obs[i][k] for i in active]) for k in obs[0]}
                with torch.inference_mode():probs=[d.probs.cpu().numpy() for d in distributions(model.policy,batch)]
                following=[]
                for j,i in enumerate(active):
                    if mode=='greedy':a=[int(p[j].argmax()) for p in probs]
                    else:
                        u=rng[i].random(4);a=[min(int(np.searchsorted(np.cumsum(p[j]),u[h])),p.shape[1]-1) for h,p in enumerate(probs)]
                    before=envs[i].pos.copy();obs[i],_,done,_,info=envs[i].step(a)
                    sort=np.sort(probs[0][j]);traces[i].append(dict(best=int(probs[0][j].argmax()),margin=float(sort[-1]-sort[-2]),action=a,delta=(envs[i].pos-before).tolist(),position=envs[i].pos.tolist()))
                    if done:results.append(dict(model_seed=seed,mode=mode,seed=5000+offset+i,success=bool(info['success']),frames=int(info['frames']),metrics=metrics(traces[i]),trace=traces[i]))
                    else:following.append(i)
                active=following
                if not active:break
            assert not active
            print(seed,mode,offset+8,flush=True)
        # Check that instrumentation reproduces saved evaluations exactly.
        ref=json.loads(pathlib.Path(f'artifacts/critic-normalization-20260930/control/{seed}/result.json').read_text())['evaluations'][mode]['episodes']
        lookup={r['seed']:r for r in ref}
        assert all(r['success']==lookup[r['seed']]['success'] and r['frames']==lookup[r['seed']]['frames'] for r in results if r['mode']==mode)
        (out/f'{seed}.json').write_text(json.dumps(results))
if __name__=='__main__':run(int(sys.argv[1]))

"""CNN-only diagnostic copies: real rollout GAE and exact minibatch updates.

Branch outcomes are labels only, never inputs or rewards. 9+2 policy unchanged.
"""
import sys, pathlib, json, copy, argparse
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from touhou_ai.autumn_ablation import GridAblation
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.simulation_speed import distributions, tune_cpu
from touhou_ai.scaled_candidate import ScaledCandidateEnv

FIELDS='pos boss frame focus dead xy vel angle birth group alive n move_angle emissions rng'.split()
class Trace(GridAblation):
    def __init__(self):
        super().__init__(); self.trace=[]
    def step(self,a):
        state={k:copy.deepcopy(getattr(self,k)) for k in FIELDS}
        result=super().step(a)
        self.trace.append((state,np.array(a),bool(self.dead)))
        return result

class TraceV4(ScaledCandidateEnv):
    def __init__(self):
        super().__init__('action_grid',death_reward=-1);self.trace=[]
    def step(self,a):
        state={k:copy.deepcopy(getattr(self,k)) for k in FIELDS}
        result=super().step(a)
        self.trace.append((state,np.array(a),bool(self.dead)))
        return result

def labels(state):
    result=[]
    for d in range(9):
        for f in range(2):
            e=GridAblation()
            for k,v in state.items():setattr(e,k,copy.deepcopy(v))
            e.observe=lambda:{}
            _,_,done,_,_=e.step([d,0,f,0]); safe2=not e.dead
            for _ in range(7):
                if done:break
                _,_,done,_,_=e.step([0,0,f,0])
            result.append([safe2,not e.dead])
    return np.array(result)

def probabilities(policy,obs):
    out=[]
    with torch.no_grad():
        for start in range(0,len(next(iter(obs.values()))),16):
            ds=distributions(policy,{k:v[start:start+16] for k,v in obs.items()})
            out.append((ds[0].probs[:,:,None]*ds[2].probs[:,None,:]).reshape(-1,18).cpu().numpy())
    return np.concatenate(out)

def run(seed,updates,out,v4=False,rollout_norm=False):
    torch.set_num_threads(1)
    env=TraceV4() if v4 else Trace()
    checkpoint=f'artifacts/critic-normalization-20260930/control/{seed}/model.zip' if v4 else f'artifacts/autumn-separate-clip-20260928/cnn-switch-{seed}.zip'
    m=SeparateClipPPO.load(checkpoint,env=env,device='cpu')
    if rollout_norm:
        from touhou_ai.rollout_advantage import RolloutAdvantagePPO
        m.__class__=RolloutAdvantagePPO
    m.set_random_seed(9000+seed); tune_cpu(m,update_threads=4 if v4 else 8)
    original_train=m.train; reports=[]
    def train():
        b=m.rollout_buffer; trace=env.trace[-m.n_steps:]
        assert len(trace)==m.n_steps and m.n_envs==1
        chosen=set(range(0,m.n_steps,32))
        for i,(_,_,dead) in enumerate(trace):
            if dead:chosen.update(range(max(0,i-7),i+1))
        ids=sorted(chosen)
        lab=np.stack([labels(trace[i][0]) for i in ids])
        acts=np.array([int(trace[i][1][0])*2+int(trace[i][1][2]) for i in ids])
        for j,i in enumerate(ids):assert bool(lab[j,acts[j],0]) == (not trace[i][2])
        obs={k:v[ids,0].copy() for k,v in b.observations.items()}
        before=probabilities(m.policy,obs)
        rows=[dict(index=i,frame=int(trace[i][0]['frame']),action=int(acts[j]),death=trace[i][2],
            advantage=float(b.advantages[i,0]),value=float(b.values[i,0]),return_=float(b.returns[i,0]),
            safe2=lab[j,:,0].tolist(),safe16=lab[j,:,1].tolist(),before=before[j].tolist()) for j,i in enumerate(ids)]
        batches=[]; pending={}; get_samples=b._get_samples; step=m.policy.optimizer.step
        verify=len(reports)==0
        if verify:
            initial=copy.deepcopy(m.policy.state_dict()); initial_opt=copy.deepcopy(m.policy.optimizer.state_dict())
            rng=np.random.get_state(); torch_rng=torch.get_rng_state(); initial_updates=m._n_updates
        def samples(indices,*args,**kwargs):
            batch=get_samples(indices,*args,**kwargs)
            adv=batch.advantages
            norm=(adv-adv.mean())/(adv.std()+1e-8) if m.normalize_advantage else adv
            pending.clear();pending.update(indices=np.array(indices),norm=norm.detach().cpu().numpy())
            return batch
        def optimizer_step(*args,**kwargs):
            p0=probabilities(m.policy,obs)
            result=step(*args,**kwargs)
            p1=probabilities(m.policy,obs)
            members={int(i):float(a) for i,a in zip(pending['indices'],pending['norm'])}
            batches.append(dict(normalized_advantages={str(i):members[i] for i in ids if i in members},
                chosen_delta=(p1[np.arange(len(ids)),acts]-p0[np.arange(len(ids)),acts]).tolist(),
                safe2_delta=((p1-p0)*lab[:,:,0]).sum(axis=1).tolist(),
                safe16_delta=((p1-p0)*lab[:,:,1]).sum(axis=1).tolist()))
            return result
        b._get_samples=samples;m.policy.optimizer.step=optimizer_step
        try:original_train()
        finally:b._get_samples=get_samples;m.policy.optimizer.step=step
        after=probabilities(m.policy,obs)
        verification=None
        if verify:
            trained=copy.deepcopy(m.policy.state_dict()); trained_opt=copy.deepcopy(m.policy.optimizer.state_dict())
            end_rng=np.random.get_state(); end_torch=torch.get_rng_state(); end_updates=m._n_updates
            m.policy.load_state_dict(initial);m.policy.optimizer.load_state_dict(initial_opt)
            np.random.set_state(rng);torch.set_rng_state(torch_rng);m._n_updates=initial_updates
            original_train()
            error=max(float((v-trained[k]).abs().max()) for k,v in m.policy.state_dict().items())
            assert error==0, f'Instrumentation changed update: {error}'
            verification={'plain_update_max_parameter_difference':error}
            m.policy.load_state_dict(trained);m.policy.optimizer.load_state_dict(trained_opt)
            np.random.set_state(end_rng);torch.set_rng_state(end_torch);m._n_updates=end_updates
        for j,r in enumerate(rows):r['after']=after[j].tolist()
        reports.append(dict(update=len(reports)+1,rows=rows,batches=batches,steps=m.n_steps,verification=verification))
        env.trace.clear()
        out.write_text(json.dumps(dict(seed=seed,updates=reports),indent=2),encoding='utf-8')
        print(seed,'update',len(reports),'probes',len(rows),'deaths',sum(r['death'] for r in rows),'batches',len(batches),flush=True)
    m.train=train
    m.learn(total_timesteps=m.n_steps*updates,reset_num_timesteps=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=7);p.add_argument('--updates',type=int,default=4)
    p.add_argument('--v4',action='store_true')
    p.add_argument('--rollout-norm',action='store_true')
    a=p.parse_args();assert not a.rollout_norm or a.v4
    out=pathlib.Path('artifacts/credit-v4-rollout-20260930' if a.rollout_norm else 'artifacts/credit-v4-20260930' if a.v4 else 'artifacts/cnn-update-credit-20260929');out.mkdir(exist_ok=True)
    run(a.seed,a.updates,out/f'{a.seed}.json',v4=a.v4,rollout_norm=a.rollout_norm)

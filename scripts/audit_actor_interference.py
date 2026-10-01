"""Counterfactual actor steps from identical weights and Adam state; offline only."""
import sys,pathlib,json,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from audit_cnn_update_credit import TraceV4,probabilities
from touhou_ai.separate_clip_ppo import SeparateClipPPO

def run(seed):
    torch.set_num_threads(2)
    env=TraceV4()
    m=SeparateClipPPO.load(f'artifacts/critic-normalization-20260930/control/{seed}/model.zip',env=env,device='cpu')
    m.set_random_seed(9000+seed)
    reports=[]
    def audit():
        b=m.rollout_buffer
        # Same first permutation and minibatch normalization as ordinary PPO.
        trace=env.trace[-512:]
        wanted=set()
        for i,(_,_,dead) in enumerate(trace):
            if dead:wanted.update(range(max(0,i-3),i+1))
        actor=list(m.policy.pi_features_extractor.parameters())+list(m.policy.mlp_extractor.policy_net.parameters())+list(m.policy.action_net.parameters())
        state=copy.deepcopy(m.policy.state_dict());opt=copy.deepcopy(m.policy.optimizer.state_dict())
        # Buffer get performs the standard flattening before yielding batches.
        batches=[];orig=b._get_samples
        def capture(ids,*a,**kw):
            batches.append(np.array(ids));return orig(ids,*a,**kw)
        b._get_samples=capture
        try:data=list(b.get(m.batch_size))
        finally:b._get_samples=orig
        for ids,d in zip(batches,data):
            adv=(d.advantages-d.advantages.mean())/(d.advantages.std()+1e-8)
            for j,i in enumerate(ids):
                if int(i) not in wanted or adv[j]>=0:continue
                obs={k:v[j:j+1] for k,v in d.observations.items()}
                action=int(d.actions[j,0])*2+int(d.actions[j,2])
                before=float(probabilities(m.policy,obs)[0,action])
                row=dict(index=int(i),death=bool(trace[i][2]),advantage=float(adv[j]),before=before,steps={})
                for mode in ['single_sgd','single_adam_fresh','single_adam_history','batch_adam_fresh','batch_adam_history','batch_entropy_adam_history']:
                    m.policy.load_state_dict(state);m.policy.optimizer.load_state_dict(copy.deepcopy(opt));m.policy.optimizer.zero_grad()
                    _,lp,ent=m.policy.evaluate_actions(d.observations,d.actions)
                    ratio=torch.exp(lp-d.old_log_prob)
                    losses=-torch.minimum(adv*ratio,adv*torch.clamp(ratio,.8,1.2))
                    loss=losses[j] if mode.startswith('single') else losses.mean()
                    if 'entropy' in mode:loss=loss-m.ent_coef*ent.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor,m.max_grad_norm)
                    if mode=='single_sgd':
                        with torch.no_grad():
                            for p in actor:
                                if p.grad is not None:p.add_(p.grad,alpha=-1e-4)
                    else:
                        if 'fresh' in mode:m.policy.optimizer.state.clear()
                        m.policy.optimizer.step()
                    row['steps'][mode]=float(probabilities(m.policy,obs)[0,action])-before
                m.policy.load_state_dict(state);m.policy.optimizer.load_state_dict(copy.deepcopy(opt))
                reports.append(row)
        assert all(torch.equal(v,state[k]) for k,v in m.policy.state_dict().items())
    m.train=audit
    m.learn(total_timesteps=512,reset_num_timesteps=True)
    out=pathlib.Path('artifacts/actor-interference-20261001');out.mkdir(exist_ok=True)
    (out/f'{seed}.json').write_text(json.dumps(dict(seed=seed,rows=reports),indent=2),encoding='utf-8')
    print(seed,len(reports),flush=True)
if __name__=='__main__':run(int(sys.argv[1]))

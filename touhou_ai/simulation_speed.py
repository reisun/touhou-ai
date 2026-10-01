"""Offline-only actor inference, reproducible batched assessment and CPU tuning."""
import numpy as np,torch
from stable_baselines3.common.policies import BaseModel

def distributions(policy,observations):
    tensor,_=policy.obs_to_tensor(observations)
    features=BaseModel.extract_features(policy,tensor,policy.pi_features_extractor)
    return policy._scheduled(policy.mlp_extractor.forward_actor(features),tensor).distribution

def assess_batch(model,cls,*,n=50,side=None,swapped=False,det=False,batch_size=16,seed=123):
    wins=0;xs=[];policy=model.policy;policy.set_training_mode(False)
    for offset in range(0,n,batch_size):
        count=min(batch_size,n-offset);envs=[cls() for _ in range(count)]
        obs=[e.reset(options={'side':side})[0] if side is not None else e.reset()[0] for e in envs]
        # Independent per-episode randomness makes batch-size changes comparable.
        rngs=[np.random.default_rng(np.random.SeedSequence([seed,0 if side is None else side+1,offset+i])) for i in range(count)]
        active=list(range(count))
        for step in range(80):
            selected=[envs[i].opposite_observation() if swapped else obs[i] for i in active]
            batch={k:np.stack([o[k] for o in selected]) for k in obs[0]}
            with torch.inference_mode():probs=[d.probs.cpu().numpy() for d in distributions(policy,batch)]
            next_active=[]
            for j,i in enumerate(active):
                u=rngs[i].random(4);action=[int(p[j].argmax()) if det else min(int(np.searchsorted(np.cumsum(p[j]),u[h])),p.shape[1]-1) for h,p in enumerate(probs)]
                obs[i],_,done,_,info=envs[i].step(action)
                if step==9:xs.append(float(envs[i].pos[0]))
                if done:wins+=info['success']
                else:next_active.append(i)
            active=next_active
            if not active:break
    return {'survival':wins/n,'x_after20f':float(np.mean(xs)) if xs else None,'samples_at20f':len(xs)}

def tune_cpu(model,update_threads=8):
    # Local instance only. Saved model class/configuration remain unchanged.
    import types
    original=model.train
    def train(self):
        before=torch.get_num_threads();torch.set_num_threads(update_threads)
        try:return original()
        finally:torch.set_num_threads(before)
    model.train=types.MethodType(train,model)
    # Bound closures must not be serialized in the checkpoint.
    original_excluded=model._excluded_save_params
    model._excluded_save_params=lambda:original_excluded()+['train','_excluded_save_params']
    torch.set_num_threads(1)
    return model

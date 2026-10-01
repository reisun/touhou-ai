"""Offline comparison of Adam first-moment memory; no live defaults changed."""
from touhou_ai.autumn_training import build_model

def build(seed):
    model,cls,config=build_model('cnn',seed)
    p=model.policy
    assert not p.optimizer.state
    assert all(g['betas']==(0.9,0.999) for g in p.optimizer.param_groups)
    for g in p.optimizer.param_groups:g['betas']=(0.5,0.999)
    p.optimizer.defaults['betas']=(0.5,0.999)
    p.optimizer_kwargs=dict(p.optimizer_kwargs,betas=(0.5,0.999))
    model.policy_kwargs=dict(model.policy_kwargs,optimizer_kwargs=p.optimizer_kwargs)
    return model,cls,config

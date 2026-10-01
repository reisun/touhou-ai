"""Fresh critic width comparisons preserving actor weights and random state."""
import math
import torch
from torch import nn
from touhou_ai.autumn_training import build_model

def build(seed):
    model,cls,config=build_model('cnn',seed)
    p=model.policy;old=p.mlp_extractor.value_net[2];head=p.value_net
    assert old.in_features==256 and old.out_features==128
    with torch.random.fork_rng(devices=[]):
        expanded=nn.Linear(256,256);output=nn.Linear(256,1)
        nn.init.orthogonal_(expanded.weight,gain=math.sqrt(2));nn.init.zeros_(expanded.bias)
        with torch.no_grad():
            expanded.weight[:128].copy_(old.weight);expanded.bias[:128].copy_(old.bias)
            output.weight.zero_();output.weight[:,:128].copy_(head.weight);output.bias.copy_(head.bias)
    p.mlp_extractor.value_net[2]=expanded;p.value_net=output
    p.mlp_extractor.latent_dim_vf=256
    arch=dict(pi=[256,128],vf=[256,256])
    p.net_arch=arch;model.policy_kwargs=dict(model.policy_kwargs,net_arch=arch)
    # This builder starts fresh; no trained optimizer state is discarded.
    assert not p.optimizer.state
    p.optimizer=type(p.optimizer)(p.parameters(),**p.optimizer.defaults)
    return model,cls,config

def build64(seed):
    model,cls,config=build_model('cnn',seed)
    p=model.policy;old=p.mlp_extractor.value_net[2];head=p.value_net
    with torch.random.fork_rng(devices=[]):
        reduced=nn.Linear(256,64);output=nn.Linear(64,1)
        with torch.no_grad():
            reduced.weight.copy_(old.weight[:64]);reduced.bias.copy_(old.bias[:64])
            output.weight.copy_(head.weight[:,:64])
            output.weight.mul_(head.weight.norm()/output.weight.norm())
            output.bias.copy_(head.bias)
    p.mlp_extractor.value_net[2]=reduced;p.value_net=output
    p.mlp_extractor.latent_dim_vf=64
    arch=dict(pi=[256,128],vf=[256,64])
    p.net_arch=arch;model.policy_kwargs=dict(model.policy_kwargs,net_arch=arch)
    assert not p.optimizer.state
    p.optimizer=type(p.optimizer)(p.parameters(),**p.optimizer.defaults)
    return model,cls,config

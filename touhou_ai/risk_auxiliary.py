"""Online observable-risk auxiliary task; targets never enter actor/critic input."""
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import CombinedExtractor
from touhou_ai.autumn_ablation import SpeedPolicy
from touhou_ai.fixed_dodge import DIRECTIONS

TARGET_KEY='risk_targets'

def observed_risk(position,bullet_positions,bullet_velocities):
    """18 binary constant-velocity predictions: move2F then stop14F.

    Uses current positions/velocities only. No future turns, spawn schedule,
    simulator branching, or trajectory history. AABB collision sampled at .25F.
    """
    origin=np.asarray(position,dtype=np.float64)
    p=np.asarray(bullet_positions,dtype=np.float64).reshape(-1,2)
    v=np.asarray(bullet_velocities,dtype=np.float64).reshape(-1,2)
    if len(p)!=len(v):raise ValueError('bullet arrays mismatch')
    near=np.linalg.norm(p-origin,axis=1)<=np.linalg.norm(v,axis=1)*16+16
    p=p[near];v=v[near]
    if not len(p):return np.zeros(18,dtype=np.float32)
    t=np.arange(.25,16.01,.25)
    velocity=np.repeat(DIRECTIONS,2,axis=0)*np.tile([4.5,2.],9)[:,None]
    player=np.clip(origin+np.minimum(t[None,:,None],2)*velocity[:,None,:],[-184,32],[184,432])
    bullet=p[None,:,:]+t[:,None,None]*v[None,:,:]
    hit=np.any(np.all(np.abs(bullet[None,:,:,:]-player[:,:,None,:])<=3,axis=-1),axis=(1,2))
    return hit.astype(np.float32)

class RiskTargets(gym.Wrapper):
    def __init__(self,cls,labels=True):
        super().__init__(cls());self.labels=labels
        self.observation_space=gym.spaces.Dict(dict(self.env.observation_space.spaces,**{TARGET_KEY:gym.spaces.Box(0,1,(18,),dtype=np.float32)}))
    @property
    def pos(self):return self.env.pos
    def augment(self,obs):
        e=self.env.unwrapped
        target=observed_risk(e.pos,e.xy[e.alive],e.vel[e.alive]) if self.labels else np.zeros(18,dtype=np.float32)
        return dict(obs,**{TARGET_KEY:target})
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);return self.augment(obs),info
    def step(self,action):
        obs,reward,done,truncated,info=self.env.step(action)
        return self.augment(obs),reward,done,truncated,info

class WithoutRiskTargets(CombinedExtractor):
    def __init__(self,observation_space):
        super().__init__(gym.spaces.Dict({k:v for k,v in observation_space.spaces.items() if k!=TARGET_KEY}))

class RiskAuxPolicy(SpeedPolicy):
    def __init__(self,*args,risk_balance='scene',**kwargs):
        if risk_balance not in ['scene','action']:raise ValueError(risk_balance)
        self.risk_balance=risk_balance
        super().__init__(*args,**kwargs)
        # The extra head does not consume randomness used by baseline action sampling.
        with torch.random.fork_rng(devices=[]):
            self.risk_head=nn.Linear(self.mlp_extractor.latent_dim_pi,18)
            nn.init.orthogonal_(self.risk_head.weight,gain=.01)
            nn.init.zeros_(self.risk_head.bias)
        lr=self.optimizer.param_groups[0]['lr']
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr,**self.optimizer_kwargs)
    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters()
        data['risk_balance']=self.risk_balance
        return data
    def evaluate_actions(self,obs,actions):
        pi,vf=self._latents(obs);dist=self._scheduled(pi,obs)
        logits=self.risk_head(pi);target=obs[TARGET_KEY]
        positive=target.sum(1);negative=18-positive
        mixed=(positive>0)&(negative>0)
        # Per-scene balanced BCE; all-safe/all-danger scenes do not teach action discrimination.
        per_scene=.5*((F.softplus(-logits)*target).sum(1)/positive.clamp_min(1)+(F.softplus(logits)*(1-target)).sum(1)/negative.clamp_min(1))
        if self.risk_balance=='scene':
            self.risk_aux_loss=per_scene[mixed].mean() if mixed.any() else logits.sum()*0
        else:
            # Balance positive/negative scenes separately for each action.
            # A constant direction-specific prior cannot improve this objective.
            count_positive=target.sum(0);count_negative=target.shape[0]-count_positive
            eligible=(count_positive>0)&(count_negative>0)
            per_action=.5*((F.softplus(-logits)*target).sum(0)/count_positive.clamp_min(1)+(F.softplus(logits)*(1-target)).sum(0)/count_negative.clamp_min(1))
            self.risk_aux_loss=per_action[eligible].mean() if eligible.any() else logits.sum()*0
        self.risk_mixed_fraction=float(mixed.float().mean().detach())
        return self.value_net(vf),dist.log_prob(actions),dist.entropy()

"""Observable short-horizon risk supplied to the learned action readout only."""
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import CombinedExtractor
from touhou_ai.autumn_ablation import SpeedPolicy
from touhou_ai.risk_auxiliary import RiskTargets,TARGET_KEY

ACTION_RISK='action_risk'

class DirectRiskInput(RiskTargets):
    def __init__(self,cls,enabled=True):
        super().__init__(cls,labels=enabled)
        spaces=dict(self.observation_space.spaces);spaces[ACTION_RISK]=spaces.pop(TARGET_KEY)
        self.observation_space=gym.spaces.Dict(spaces)
    def augment(self,obs):
        result=super().augment(obs);result[ACTION_RISK]=result.pop(TARGET_KEY)
        return result

class WithoutActionRisk(CombinedExtractor):
    def __init__(self,observation_space):
        super().__init__(gym.spaces.Dict({k:v for k,v in observation_space.spaces.items() if k!=ACTION_RISK}))

class DirectRiskPolicy(SpeedPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original=self.action_net
        # Preserve original weights and initial action probabilities exactly.
        # Extra risk coefficients start at zero and are learned by PPO rewards.
        with torch.random.fork_rng(devices=[]):
            extended=nn.Linear(original.in_features+18,original.out_features)
        with torch.no_grad():
            extended.weight[:,:original.in_features].copy_(original.weight)
            extended.weight[:,original.in_features:].zero_()
            extended.bias.copy_(original.bias)
        self.action_net=extended
        lr=self.optimizer.param_groups[0]['lr']
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr,**self.optimizer_kwargs)
    def _scheduled(self,latent,obs):
        return super()._scheduled(torch.cat([latent,obs[ACTION_RISK]],dim=1),obs)

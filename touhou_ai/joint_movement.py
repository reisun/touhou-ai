"""Joint direction/focus policy for offline, shot/bomb-disabled comparisons."""
import torch
from torch import nn
from torch.distributions import Categorical
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from touhou_ai.autumn_ablation import SpeedPolicy

class JointMovementDistribution(MultiCategoricalDistribution):
 def __init__(self,logits):
  super().__init__([9,2,2,2])
  self.joint=Categorical(logits=logits)
  # One factor: legacy four-factor evaluators must not silently use marginals.
  self.distribution=[self.joint]
 @staticmethod
 def decode(index):
  zero=torch.zeros_like(index)
  return torch.stack((index//2,zero,index%2,zero),dim=-1)
 def sample(self):return self.decode(self.joint.sample())
 def mode(self):return self.decode(self.joint.probs.argmax(dim=-1))
 def log_prob(self,actions):
  return self.joint.log_prob(actions[:,0].long()*2+actions[:,2].long())
 def entropy(self):return self.joint.entropy()

class JointMovementPolicy(SpeedPolicy):
 def __init__(self,*args,**kwargs):
  if kwargs.get('fixed_focus') is not None:raise ValueError('joint experiment requires speed switching')
  super().__init__(*args,**kwargs)
 def _build(self,lr_schedule):
  # Retain identical initial encoder/critic weights to the baseline for a given seed.
  super()._build(lr_schedule)
  self.action_net=nn.Linear(self.mlp_extractor.latent_dim_pi,18)
  if self.ortho_init:self.action_net.apply(lambda module:self.init_weights(module,gain=.01))
  self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
 def _scheduled(self,latent,obs):
  return JointMovementDistribution(self.action_net(latent))

"""Offline-only selected movement history ablation; no live collector changes."""
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from scripts.live_jitter_trial import Env
from touhou_ai.fixed_dodge import DIRECTIONS
from touhou_ai.spatial_input_candidates import CandidateFeatures
from touhou_ai.scaled_candidate import ScaledCandidateEnv
from touhou_ai.focused_policy import reward_input


class HistoryEnv(Env):
    def __init__(self, penalty=.1, history=True):
        self.history_enabled=history
        self.history=np.zeros((6,4),dtype=np.float32)
        super().__init__(penalty)
        self.observation_space=gym.spaces.Dict(dict(self.observation_space.spaces)|{
            'movement_history':gym.spaces.Box(-1,1,(6,4),dtype=np.float32)})

    def reset(self, **kwargs):
        self.history.fill(0)
        return super().reset(**kwargs)

    def observe(self):
        obs=super().observe()
        obs['movement_history']=self.history.copy() if self.history_enabled else np.zeros_like(self.history)
        return obs

    def step(self, action):
        # Physics encodes the scene once. Updating reward/history needs no re-rasterization.
        p=self.pos.copy()
        obs,reward,done,truncated,info=ScaledCandidateEnv.step(self,action)
        self.moves.append(self.pos-p);self.moves=self.moves[-6:]
        self.cooldown=max(0,self.cooldown-1)
        flagged=False
        if len(self.moves)==6 and not self.cooldown:
            v=np.array(self.moves);norm=np.linalg.norm(v,axis=1);length=norm.sum()
            rev=(np.sum(v[1:]*v[:-1],axis=1)<-.5*norm[1:]*norm[:-1])&(norm[1:]>.01)&(norm[:-1]>.01)
            flagged=bool(length>=1 and 1-np.linalg.norm(v.sum(0))/length>=.75 and rev.sum()>=2)
            if flagged:self.cooldown=6
        self.previous={'hit':reward}
        self.history[:-1]=self.history[1:]
        self.history[-1]=[*DIRECTIONS[int(action[0])],int(action[2]),1.]
        obs['previous_rewards']=reward_input(self.previous)
        obs['movement_history']=self.history.copy() if self.history_enabled else np.zeros_like(self.history)
        return obs,reward-self.penalty*flagged,done,truncated,dict(info,jitter=flagged,penalty=self.penalty*flagged)


class HistoryFeatures(CandidateFeatures):
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self.merge[0]=nn.Linear(353+24,256)

    def forward(self, obs):
        parts=[self.local(obs['local_grid']),self.global_scene(obs['global_grid']),
               self.player(obs['player']),self.reward(obs['previous_rewards']),
               obs['bomb_clock'],self.action(obs['action_grid']),obs['movement_history'].flatten(1)]
        return self.merge(torch.cat(parts,dim=1))

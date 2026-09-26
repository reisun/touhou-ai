"""Versioned local-bullet architecture and previous-transition reward inputs."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

CONTRACT = 'th10-focused-bullets-v2'
REWARD_KEYS = ('damage', 'progress', 'hit', 'power_down')
# Fixed scales are independent of configurable reward weights. Bounded transform
# preserves sign and avoids clipping distinct large rewards to the same value.
REWARD_SCALES = np.asarray([1., 100., 10., 1.], dtype=np.float32)


def reward_input(components=None):
    components = components or {}
    if set(components) - set(REWARD_KEYS):
        raise ValueError('unknown reward input component')
    values = np.asarray([components.get(k, 0.) for k in REWARD_KEYS], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite reward input')
    scaled = values / REWARD_SCALES
    return scaled / (1. + np.abs(scaled))


class FocusedFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        self.entities = nn.ModuleDict()
        for key in ('bullets', 'enemies', 'items', 'lasers'):
            width = 64 if key == 'bullets' else 32
            self.entities['set_' + key] = nn.Sequential(
                nn.Linear(observation_space[key].shape[1]-1, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU())
        self.near = nn.Sequential(nn.Flatten(), nn.Linear(80, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU())
        self.player = nn.Sequential(nn.Linear(21, 32), nn.ReLU())
        self.reward = nn.Sequential(nn.Linear(len(REWARD_KEYS), 32), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(513, 256), nn.ReLU())

    def forward(self, observations):
        parts = [self.player(observations['player']), self.reward(observations['previous_rewards'])]
        near = observations['near_bullets']
        parts.append(self.near(near * near[:, :, :1]))
        for key in ('bullets', 'enemies', 'items', 'lasers'):
            entity = observations[key]
            mask = entity[:, :, :1]
            encoded = self.entities['set_' + key](entity[:, :, 1:]) * mask
            parts.extend([encoded.sum(1)/mask.sum(1).clamp(min=1), encoded.max(1).values])
        return self.merge(torch.cat([*parts, observations['bomb_clock']], dim=1))

"""Scheduled bomb factor: 12F decisions, 2F pulse, 10F release.

The clock is stored in observations so PPO minibatches and GPU workers use the
same probability measure as collection. Nondecision bomb actions are not draws.
"""
import math
import numpy as np
import torch
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

INITIAL_PROBABILITY = 1 - 0.5 ** (1 / 25)
SPEC = {'version': 1, 'frames_per_decision': 12, 'other_action_frames': 2,
        'press_frames': 2, 'release_frames': 10,
        'initial_probability': INITIAL_PROBABILITY,
        'availability_mask': False, 'nondecision_log_prob_entropy': 0}


class BombClock:
    def __init__(self):
        self.frames = 0

    def encode(self):
        return np.asarray([(self.frames % 12) / 12], dtype=np.float32)

    def advance(self, frames):
        if type(frames) is not int or frames <= 0 or frames % 2:
            raise ValueError('bomb schedule requires positive even gameplay frames')
        self.frames += frames


class TimedDistribution(MultiCategoricalDistribution):
    def __init__(self, base, due):
        super().__init__(base.action_dims)
        self.distribution = base.distribution
        self.due = due

    def log_prob(self, actions):
        terms = [d.log_prob(actions[:, i]) for i, d in enumerate(self.distribution)]
        return sum(terms[:3]) + terms[3] * self.due

    def entropy(self):
        terms = [d.entropy() for d in self.distribution]
        return sum(terms[:3]) + terms[3] * self.due

    def sample(self):
        actions = [d.sample() for d in self.distribution[:3]]
        bomb = torch.zeros_like(actions[0])
        selected = self.due.bool()
        if selected.any():
            bomb[selected] = torch.distributions.Categorical(
                logits=self.distribution[3].logits[selected]).sample()
        return torch.stack([*actions, bomb], dim=1)

    def mode(self):
        actions = super().mode()
        actions[:, 3] *= self.due.long()
        return actions


class TimedBombPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if list(self.action_space.nvec) != [9, 2, 2, 2] or 'bomb_clock' not in self.observation_space.spaces:
            raise ValueError('timed bomb policy requires scheduled four-head observations')
        with torch.no_grad():
            self.action_net.weight[-2:].zero_()
            self.action_net.bias[-2] = 0
            self.action_net.bias[-1] = math.log(INITIAL_PROBABILITY / (1 - INITIAL_PROBABILITY))

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        return (self.mlp_extractor.forward_actor(features[0]),
                self.mlp_extractor.forward_critic(features[1]))

    def _scheduled(self, latent, obs):
        base = self._get_action_dist_from_latent(latent)
        return TimedDistribution(base, (obs['bomb_clock'][:, 0] == 0).to(latent.dtype))

    def forward(self, obs, deterministic=False):
        pi, vf = self._latents(obs)
        dist = self._scheduled(pi, obs)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self.value_net(vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        pi, vf = self._latents(obs)
        dist = self._scheduled(pi, obs)
        return self.value_net(vf), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        pi, _ = self._latents(obs)
        return self._scheduled(pi, obs)

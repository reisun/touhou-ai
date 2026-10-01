"""Offline forward-view observation comparison."""
import functools
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from touhou_ai.dual_grid import DualGridContract, GridRolloutBuffer, paint_laser
from touhou_ai.spatial_input_candidates import CandidateFeatures, bullet_arrays, paint, risk_grid
from touhou_ai.scaled_candidate import ScaledCandidateEnv

VARIANTS = ('control', 'detail', 'wide', 'both')


class ForwardEncoder:
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.front = 144 if variant in ('detail', 'both') else 96
        self.wide = variant in ('wide', 'both')
        self.base = DualGridContract()
        spaces = dict(self.base.observation_space.spaces)
        spaces['action_grid'] = gym.spaces.Box(0, 1, (4, 3, 3), dtype=np.float32)
        if self.wide:
            original = spaces['global_grid']
            spaces['global_grid'] = gym.spaces.Box(original.low[3:], original.high[3:], dtype=np.float32)
            spaces['wide_grid'] = gym.spaces.Box(0, 1, (3, 64, 64), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def encode(self, raw):
        if self.variant == 'control':
            result = self.base.encode(raw)
        else:
            result = self.base.encode(dict(raw, bullets=[], lasers=[]))
            position = np.asarray(raw['player']['position'], dtype=float)
            origin = position - [96, self.front]
            positions, velocities, sizes = bullet_arrays(raw)
            local = np.zeros((6, 96, 96), dtype=np.float32)
            paint(local[0], position[None], np.asarray(raw['player']['hitbox_raw'])[None], origin, 2)
            for channel, frames in enumerate((0, 2, 4), 1):
                paint(local[channel], positions + frames * velocities, sizes / 2, origin, 2)
            for laser in raw['lasers']:
                paint_laser(local, 4, 5, laser['collision'], origin, 2)
                paint_laser(result['global_grid'], 8, 9, laser['collision'], np.array([-192, 0]), 8)
            result['local_grid'] = local
            if self.wide:
                result['global_grid'] = result['global_grid'][3:]
                wide = np.zeros((3, 64, 64), dtype=np.float32)
                wide_origin = position - [128, self.front + 256]
                for channel, frames in enumerate((0, 2, 4)):
                    paint(wide[channel], positions + frames * velocities, sizes / 2, wide_origin, 4)
                result['wide_grid'] = wide
            else:
                from touhou_ai.dual_grid import bin_entities
                active = [bullet for bullet in raw['bullets'] if bullet['flags_raw'] & 2]
                for channel, frames in enumerate((0, 2, 4)):
                    bin_entities(active, (channel,), result['global_grid'], frames)
        result['action_grid'] = risk_grid(raw)
        for key, value in result.items():
            space = self.observation_space.spaces[key]
            np.clip(value, space.low, space.high, out=value)
            if key.endswith('_grid'):
                result[key] = value.astype(np.float16).astype(np.float32)
        return result


class ForwardEnv(ScaledCandidateEnv):
    def __init__(self, variant='control'):
        super().__init__()
        self.encoder = ForwardEncoder(variant)
        self.observation_space = self.encoder.observation_space


class ForwardFeatures(CandidateFeatures):
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self.has_wide = 'wide_grid' in observation_space.spaces
        if self.has_wide:
            self.global_scene[0] = nn.Conv2d(9, 4, 3, padding=1)
            self.wide = nn.Sequential(
                nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(4 * 16 * 16, 64), nn.ReLU())
            self.merge[0] = nn.Linear(417, 256)

    def forward(self, observations):
        if not self.has_wide:
            return super().forward(observations)
        parts = [self.local(observations['local_grid']), self.global_scene(observations['global_grid']),
                 self.player(observations['player']), self.reward(observations['previous_rewards']),
                 observations['bomb_clock'], self.action(observations['action_grid']),
                 self.wide(observations['wide_grid'])]
        return self.merge(torch.cat(parts, dim=1))


def build(variant, seed):
    from touhou_ai.autumn_training import build_model
    from touhou_ai.autumn_ablation import SpeedPolicy
    from touhou_ai.separate_clip_ppo import SeparateClipPPO
    baseline, _, config = build_model('cnn', seed)
    environment = functools.partial(ForwardEnv, variant=variant)
    if variant in ('control', 'detail'):
        baseline.set_env(environment())
        baseline.env.seed(config['training_environment_initial_seed'])
        return baseline, environment, config
    numpy_state = np.random.get_state()
    with torch.random.fork_rng():
        model = SeparateClipPPO(SpeedPolicy, environment(), seed=seed, device='cpu',
            rollout_buffer_class=GridRolloutBuffer, verbose=0,
            policy_kwargs=dict(share_features_extractor=False, net_arch=dict(pi=[256, 128], vf=[256, 128]),
                               features_extractor_class=ForwardFeatures), **config['ppo'])
        original = baseline.policy.state_dict()
        with torch.no_grad():
            for key, target in model.policy.state_dict().items():
                if key not in original:
                    continue
                source = original[key]
                if target.shape == source.shape:
                    target.copy_(source)
                elif key.endswith('global_scene.0.weight'):
                    target.copy_(source[:, 3:])
                elif key.endswith('merge.0.weight'):
                    target.zero_()
                    target[:, :source.shape[1]].copy_(source)
                else:
                    raise ValueError(key)
    np.random.set_state(numpy_state)
    model.env.seed(config['training_environment_initial_seed'])
    return model, environment, config

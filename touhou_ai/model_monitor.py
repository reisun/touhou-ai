"""Read-only inference on actual observations; never owns a game input device."""
import json
from touhou_ai.bullet_scope import SPEC as BULLET_SCOPE


class NoCompatibleCheckpoint(ValueError):
    pass


def model_metadata(model, contract, checkpoint, migrated=False, reward_weights=None):
    keys = ('seed', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range',
            'ent_coef', 'n_steps', 'batch_size', 'n_epochs')
    settings = {}
    for key in keys:
        value = getattr(model, key)
        settings[key] = value(1.0) if callable(value) else value
    from touhou_ai.focused_policy import REWARD_KEYS, REWARD_SCALES
    from touhou_ai.dual_grid import CONTRACT as GRID_CONTRACT, SPEC as GRID_SPEC
    from touhou_ai.timed_bomb import SPEC as BOMB_SCHEDULE
    from touhou_ai.learning_discount import SPEC as DISCOUNT_CONTRACT
    return {'contract': contract, 'checkpoint': checkpoint, 'migrated': migrated,
            'bullet_scope': GRID_SPEC if contract == GRID_CONTRACT else BULLET_SCOPE,
            'reward_input': {'keys': list(REWARD_KEYS), 'scales': REWARD_SCALES.tolist(),
                             'transform': '(value/scale)/(1+abs(value/scale))',
                             'timing': 'previous_completed_action'} if 'previous_rewards' in model.observation_space.spaces else None,
            'new_features_trained': not migrated,
            'observation_shapes': {k: list(v.shape) for k, v in model.observation_space.spaces.items()},
            'action_heads': model.action_space.nvec.tolist(), 'ppo': settings,
            'discount_contract': DISCOUNT_CONTRACT if settings['gamma'] == DISCOUNT_CONTRACT['gamma'] else None,
            'bomb_schedule': BOMB_SCHEDULE if 'bomb_clock' in model.observation_space.spaces else None,
            'rollout_boundary': 'game_over', 'reward_weights': reward_weights or {},
            'steps': model.num_timesteps}


class ModelMonitor:
    def __init__(self):
        import torch
        from stable_baselines3 import PPO
        from touhou_ai.live_learning import ROOT, resume_manifest, FocusedObservedContract
        from touhou_ai.focused_policy import CONTRACT as EXTENDED_CONTRACT
        from touhou_ai.live_rewards import VERSION, WEIGHTS, ENABLED
        torch.set_num_threads(1)
        candidates = []
        for path in (ROOT / 'artifacts').glob('live-learning-*/status.json'):
            status = json.loads(path.read_text(encoding='utf-8'))
            if (status.get('backend') != 'real_th10' or status.get('reward_version') != VERSION
                    or status.get('bullet_scope') != BULLET_SCOPE
                    or status.get('contract') != EXTENDED_CONTRACT):
                continue
            for episode in status.get('episodes', []):
                if episode.get('reload_verified'):
                    candidates.append((episode['total_steps'], path.parent / episode['checkpoint']))
        if not candidates:
            raise NoCompatibleCheckpoint('no verified real-game checkpoint for current contracts')
        checkpoint = max(candidates, key=lambda pair: (pair[0], str(pair[1])))[1]
        manifest, episode = resume_manifest(checkpoint, extended=True)
        old = PPO.load(checkpoint, device='cpu')
        if old.num_timesteps != episode['total_steps']:
            raise ValueError('checkpoint timestep mismatch')
        self.env = FocusedObservedContract()
        migrated = False
        self.model = old
        self.model.policy.set_training_mode(False)
        profile = json.loads((ROOT / 'configs/sharu-inspired-v1.json').read_text(encoding='utf-8'))
        self.metadata = model_metadata(self.model, EXTENDED_CONTRACT,
            str(checkpoint.relative_to(ROOT)), migrated,
            WEIGHTS)
        self.metadata['reward_enabled'] = ENABLED
        self.metadata['checkpoint_contract'] = manifest['contract']
        self.metadata['checkpoint_updates'] = episode['total_updates']
        self.metadata['reward_scope'] = 'next_extended_training'

    def attach(self, raw, data):
        from touhou_ai.telemetry import policy_packet
        data['model'] = self.metadata
        data['policy'] = None
        data['capabilities']['policy_connected'] = False
        # Passive observation cannot reconstruct exact action-level combat rewards.
        # Do not silently infer with fabricated zero history for this contract.
        if 'previous_rewards' in self.env.observation_space.spaces:
            data['policy_unavailable'] = 'previous_action_reward_history_requires_live_collector'
            return
        if raw.get('player') is None or raw.get('replay_mode') != 0 or raw.get('mode_flags') != 0:
            data['policy_unavailable'] = 'not_in_gameplay'
            return
        try:
            observation = self.env.encode(raw)
            policy = policy_packet(self.model, observation)
            policy.update(source='real_observation_shadow', trained=True,
                          input_control=False, observation_frame=raw['stage_frame'],
                          trained_updates=self.metadata['checkpoint_updates'],
                          new_features_trained=self.metadata['new_features_trained'])
            data['policy'] = policy
            data['capabilities']['policy_connected'] = True
        except ValueError as error:
            data['policy_unavailable'] = str(error)

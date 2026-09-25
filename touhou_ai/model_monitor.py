"""Read-only inference on actual observations; never owns a game input device."""
import json


def model_metadata(model, contract, checkpoint, migrated=False, reward_weights=None):
    keys = ('seed', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range',
            'ent_coef', 'n_steps', 'batch_size', 'n_epochs')
    settings = {}
    for key in keys:
        value = getattr(model, key)
        settings[key] = value(1.0) if callable(value) else value
    return {'contract': contract, 'checkpoint': checkpoint, 'migrated': migrated,
            'new_features_trained': not migrated,
            'observation_shapes': {k: list(v.shape) for k, v in model.observation_space.spaces.items()},
            'action_heads': model.action_space.nvec.tolist(), 'ppo': settings,
            'rollout_boundary': 'game_over', 'reward_weights': reward_weights or {},
            'steps': model.num_timesteps}


class ModelMonitor:
    def __init__(self):
        import torch
        from stable_baselines3 import PPO
        from touhou_ai.live_learning import ROOT, resume_manifest, ExtendedObservedContract, migrate_extended
        from touhou_ai.live_features import EXTENDED_CONTRACT
        torch.set_num_threads(1)
        candidates = []
        for path in (ROOT / 'artifacts').glob('live-learning-*/status.json'):
            status = json.loads(path.read_text(encoding='utf-8'))
            if status.get('backend') != 'real_th10':
                continue
            for episode in status.get('episodes', []):
                if episode.get('reload_verified'):
                    candidates.append((episode['total_steps'], path.parent / episode['checkpoint']))
        if not candidates:
            raise ValueError('no verified real-game checkpoint')
        checkpoint = max(candidates, key=lambda pair: (pair[0], str(pair[1])))[1]
        manifest, episode = resume_manifest(checkpoint, extended=True)
        old = PPO.load(checkpoint, device='cpu')
        if old.num_timesteps != episode['total_steps']:
            raise ValueError('checkpoint timestep mismatch')
        self.env = ExtendedObservedContract()
        migrated = manifest['contract'] != EXTENDED_CONTRACT
        self.model = (migrate_extended(old, PPO('MultiInputPolicy', self.env, device='cpu',
            **manifest['configured_ppo'], policy_kwargs=old.policy_kwargs)) if migrated else old)
        self.model.policy.set_training_mode(False)
        profile = json.loads((ROOT / 'configs/sharu-inspired-v1.json').read_text(encoding='utf-8'))
        self.metadata = model_metadata(self.model, EXTENDED_CONTRACT,
            str(checkpoint.relative_to(ROOT)), migrated,
            {k: profile['provisional_reward_weights'][k] for k in ('hit', 'bomb')})
        self.metadata['checkpoint_contract'] = manifest['contract']
        self.metadata['checkpoint_updates'] = episode['total_updates']
        self.metadata['reward_scope'] = 'next_extended_training'

    def attach(self, raw, data):
        from touhou_ai.telemetry import policy_packet
        data['model'] = self.metadata
        data['policy'] = None
        data['capabilities']['policy_connected'] = False
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

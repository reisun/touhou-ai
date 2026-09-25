import copy
import math
import unittest

from touhou_ai.live_features import bomb_events, laser_collision
from touhou_ai.th10_reader import Th10Reader


class ExtendedFeatureTests(unittest.TestCase):
    def test_bomb_request_is_not_activation_reward(self):
        before = {'bomb': {'state': 0}, 'stage_frame': 1}
        self.assertEqual(bomb_events(before, before | {'input_state_raw': [2]}), [])
        active = {'bomb': {'state': 1}, 'stage_frame': 3}
        self.assertEqual(bomb_events(before, active)[0]['kind'], 'bomb')
        self.assertEqual(bomb_events(active, active), [])
        self.assertEqual(bomb_events(active, before), [])
        with self.assertRaises(ValueError):
            bomb_events(before, before | {'bomb': {'state': 2}})

    def test_acceleration_requires_continuous_lifetime_and_frames(self):
        reader = Th10Reader(None)
        def sample(frame, age, vx, manager=1):
            state = {'stage': 1, 'stage_frame': frame, 'bullets': [
                {'slot': 1, 'age_frames': age, 'type': 2, 'status': 1, 'velocity_raw': [vx, 1]}]}
            reader.add_acceleration(state, manager)
            return state['bullets'][0]['acceleration']
        self.assertIsNone(sample(10, 5, 1))
        self.assertEqual(sample(12, 7, 2), [.5, 0])
        self.assertEqual(sample(12, 7, 2), [.5, 0])
        self.assertIsNone(sample(14, 1, 3))
        self.assertIsNone(sample(20, 7, 4))
        self.assertIsNone(sample(22, 9, 5, manager=2))
        self.assertIsNone(sample(1, 1, 5, manager=2))

    def test_laser_rectangles_exclude_telegraph_and_shrink(self):
        line = laser_collision('line', 0, [10, 20], 0, 100, 16)
        self.assertEqual(line['origin'], [20, 20])
        self.assertEqual((line['length'], line['width']), (80, 8))
        self.assertTrue(line['active'])
        self.assertFalse(line['field_validated'])
        self.assertFalse(laser_collision('line', 0, [0, 0], 0, 100, 3)['active'])
        for state in (1, 3, 5):
            self.assertFalse(laser_collision('infinite', state, [0, 0], 0, 100, 16)['active'])
        for state in (2, 4):
            self.assertTrue(laser_collision('infinite', state, [0, 0], 0, 100, 16)['active'])
        self.assertIsNone(laser_collision('unknown', 0, [0, 0], 0, 1, 1))
        with self.assertRaises(ValueError):
            laser_collision('line', 0, [0, 0], math.nan, 1, 1)

    def test_extended_values_reach_model_and_migration_preserves_old_heads(self):
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from touhou_ai.live_learning import ObservedContract, ExtendedObservedContract, migrate_extended
        from touhou_ai.policy_check import NumericalFeatures
        torch.set_num_threads(1)
        old_env, new_env = ObservedContract(), ExtendedObservedContract()
        raw = {'stage': 1, 'stage_frame': 10, 'lives_raw': 2, 'power_raw': 40,
               'player': {'position': [0, 400], 'velocity_raw': [0, 0], 'status': 1,
                          'invincibility_raw': 0, 'focus_raw': 0},
               'bomb': {'state': 0}, 'spell': {'flags_raw': 1, 'id_raw': 3},
               'bullets': [{'position': [10, 100], 'velocity_raw': [1, 2], 'acceleration': [.25, -.5]}],
               'enemies': [{'position': [0, 80], 'velocity_raw': [0, 0], 'hp': 2000, 'hp_max': 8000, 'is_boss': True}],
               'items': [], 'lasers': []}
        observation = new_env.encode(raw)
        self.assertTrue(new_env.observation_space.contains(observation))
        np.testing.assert_array_equal(observation['bullets'][0, 5:], [.25, -.5, 1])
        self.assertEqual(observation['enemies'][0, 5], .25)
        self.assertEqual(observation['player'][17], 1)
        self.assertTrue(new_env.action_space.contains([0, 0, 0, 1]))
        settings = dict(n_steps=8, batch_size=8, n_epochs=1, device='cpu',
                        policy_kwargs={'features_extractor_class': NumericalFeatures,
                                       'net_arch': {'pi': [128, 128], 'vf': [128, 128]}})
        old = PPO('MultiInputPolicy', old_env, **settings)
        new = migrate_extended(old, PPO('MultiInputPolicy', new_env, **settings))
        with torch.no_grad():
            old_tensor = old.policy.obs_to_tensor(old_env.encode(raw))[0]
            new_tensor = new.policy.obs_to_tensor(observation)[0]
            old_heads = old.policy.get_distribution(old_tensor).distribution
            new_heads = new.policy.get_distribution(new_tensor).distribution
            for a, b in zip(old_heads[:3], new_heads[:3]):
                torch.testing.assert_close(a.probs, b.probs)
            torch.testing.assert_close(old.policy.predict_values(old_tensor), new.policy.predict_values(new_tensor))
            self.assertLess(new_heads[3].probs[0, 1], .02)
        other = copy.deepcopy(raw)
        other['bullets'][0]['acceleration'] = None
        self.assertEqual(new_env.encode(other)['bullets'][0, 7], 0)
        other['lasers'] = [{'position': [10, 100], 'velocity_raw': [1, 0],
                            'collision': laser_collision('line', 2, [10, 100], 0, 100, 14)}]
        encoded = new_env.encode(other)
        self.assertAlmostEqual(float(encoded['lasers'][0, 1]), 20/384)
        self.assertAlmostEqual(float(encoded['lasers'][0, 7]), 80/448)
        self.assertEqual(encoded['lasers'][0, 11], 0)
        from touhou_ai.model_monitor import model_metadata, ModelMonitor
        monitor = ModelMonitor.__new__(ModelMonitor)
        monitor.env, monitor.model = new_env, new
        monitor.metadata = model_metadata(new, 'th10-live-observed-v2', 'local', True, {'hit': -5})
        monitor.metadata['checkpoint_updates'] = 0
        data = {'capabilities': {}}
        monitor.attach(other | {'replay_mode': 0, 'mode_flags': 0}, data)
        self.assertEqual(data['policy']['source'], 'real_observation_shadow')
        self.assertFalse(data['policy']['input_control'])
        self.assertFalse(data['policy']['new_features_trained'])
        self.assertEqual(data['model']['observation_shapes']['lasers'], [64, 12])
        monitor.attach(other | {'replay_mode': 1}, data)
        self.assertIsNone(data['policy'])
        self.assertFalse(data['capabilities']['policy_connected'])

    def test_dialogue_is_confirmed_outside_policy_and_rejects_pause(self):
        from touhou_ai.live_reset import advance_dialogue
        class Fake:
            commands = []
            state = {'dialogue_raw': 1, 'player': {}, 'replay_mode': 0, 'mode_flags': 0,
                     'lives_raw': 2, 'pause_words': [2, 0]}
            def snapshot(self, full=True):
                return self.state
            def step(self, mask, frames, full=False):
                self.commands.append(mask)
                self.state = self.state | {'dialogue_raw': 0}
                return self.state
        runtime = Fake()
        self.assertFalse(advance_dialogue(runtime)['dialogue_raw'])
        self.assertEqual(runtime.commands, [1])
        runtime.state = runtime.state | {'dialogue_raw': 1, 'pause_words': [2, 2]}
        with self.assertRaises(ValueError):
            advance_dialogue(runtime)

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

from touhou_ai.dual_grid import (CONTRACT, SPEC, DualGridContract, DualGridFeatures,
                                 GridRolloutBuffer, rectangle)
from touhou_ai.live_learning import finish_buffer, resume_manifest
from touhou_ai.model_monitor import model_metadata


def state():
    return {'stage': 1, 'lives_raw': 2, 'power_raw': 40,
            'player': {'position': [0, 300], 'velocity_raw': [0, 0], 'hitbox_raw': [1, 1],
                       'status': 1, 'invincibility_raw': 0, 'focus_raw': 0},
            'bullets': [], 'enemies': [], 'items': [], 'lasers': [],
            'bomb': {'state': 0}, 'spell': None}


def bullet(x=0, y=300, vx=2, vy=-1):
    return {'position': [x, y], 'velocity_raw': [vx, vy], 'hitbox_raw': [4, 4],
            'flags_raw': 2, 'status': 1}


def enemy(x, y, hp=50):
    return {'position': [x, y], 'velocity_raw': [1, 0], 'hp': hp, 'hp_max': 100,
            'is_boss': False}


class DualGridTests(unittest.TestCase):
    def test_fractional_coverage_and_extent_intersecting_from_outside(self):
        env = DualGridContract(); raw = state()
        raw['bullets'] = [bullet(), bullet(97, 300)]
        obs = env.encode(raw)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(set(obs), {'local_grid', 'global_grid', 'player', 'previous_rewards', 'bomb_clock'})
        # Player half-size 1 => 2x2 area, four quarter-occupied 2px cells.
        np.testing.assert_array_equal(obs['local_grid'][0, 47:49, 47:49], .25)
        self.assertEqual(obs['local_grid'][0].sum(), 1)
        # Bullet size 4 is full width, giving four completely occupied cells.
        np.testing.assert_array_equal(obs['local_grid'][1, 47:49, 47:49], 1)
        # Center is outside the 96px radius, but hitbox overlaps the last column.
        np.testing.assert_array_equal(obs['local_grid'][1, 47:49, 95], .5)
        np.testing.assert_allclose(obs['local_grid'][2, 47:49, 47:49], .2, atol=.0001)
        shifted = copy.deepcopy(raw)
        for entity in [shifted['player'], *shifted['bullets']]:
            entity['position'][0] += 13.25; entity['position'][1] -= 5.5
        np.testing.assert_array_equal(obs['local_grid'], env.encode(shifted)['local_grid'])

    def test_all_entities_global_overlap_and_enemy_attributes(self):
        raw = state()
        raw['bullets'] = [bullet(-180+i*8, 40) for i in range(45)] + [bullet()]
        raw['enemies'] = [enemy(-180+i*8, 80) for i in range(30)]
        raw['enemies'][-1]['is_boss'] = True
        raw['items'] = [{'position': [-180+i*8, 100], 'velocity_raw': [0, 1]} for i in range(45)]
        obs = DualGridContract().encode(raw, {'damage': 1})
        whole = obs['global_grid']
        self.assertEqual(np.count_nonzero(whole[0]), 46)
        self.assertEqual(np.count_nonzero(whole[3]), 30)
        self.assertEqual(np.count_nonzero(whole[10]), 45)
        np.testing.assert_array_equal(whole[6][whole[3] > 0], .5)
        self.assertEqual(np.count_nonzero(whole[9]), 1)
        self.assertGreater(whole[0, 37, 24], 0)  # Near bullet remains in global view.
        self.assertEqual(obs['previous_rewards'][0], .5)
        self.assertEqual(DualGridContract().encode(raw)['previous_rewards'].sum(), 0)

    def test_inactive_missing_unknown_hp_and_opposed_velocities(self):
        env = DualGridContract(); raw = state()
        raw['bullets'] = [bullet(vx=4), bullet(vx=-4), bullet(30, 300) | {'flags_raw': 0}]
        obs = env.encode(raw)
        self.assertEqual(obs['local_grid'][2].sum(), 0)
        self.assertEqual(np.count_nonzero(obs['global_grid'][0]), 1)
        raw['enemies'] = [enemy(0, 200) | {'hp': None}, enemy(0, 200)]
        obs = env.encode(raw)
        self.assertEqual(obs['global_grid'][6, 25, 24], .5)
        self.assertEqual(obs['global_grid'][7, 25, 24], .5)
        with self.assertRaisesRegex(ValueError, 'flags'):
            bad = copy.deepcopy(raw); del bad['bullets'][0]['flags_raw']; env.encode(bad)
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            env.encode(raw | {'bullets': None})
        with self.assertRaisesRegex(ValueError, 'hitbox'):
            bad = copy.deepcopy(raw); bad['bullets'][0]['hitbox_raw'] = [-2, 4]; env.encode(bad)
        with self.assertRaisesRegex(ValueError, 'velocity'):
            bad = copy.deepcopy(raw); bad['bullets'][0]['velocity_raw'] = [float('nan'), 0]; env.encode(bad)

    def test_viewport_boundaries_and_laser_layers(self):
        self.assertIsNone(rectangle(np.array([1000, 1000]), np.array([2, 2]), np.array([0, 0]), 2, (96, 96)))
        raw = state()
        raw['bullets'] = [bullet(-193, 5), bullet(192, 5), bullet(-192, 0)]
        collision = {'origin': [-10, 300], 'angle': 0, 'length': 20, 'width': 4,
                     'active': True, 'field_validated': True}
        raw['lasers'] = [{'collision': collision}]
        obs = DualGridContract().encode(raw)
        self.assertEqual(np.count_nonzero(obs['global_grid'][0]), 1)
        self.assertEqual(obs['local_grid'][4].sum(), 20)
        np.testing.assert_array_equal(obs['local_grid'][4], obs['local_grid'][5])
        raw['lasers'][0]['collision']['field_validated'] = False
        self.assertEqual(DualGridContract().encode(raw)['local_grid'][5].sum(), 0)
        raw['lasers'][0]['collision']['active'] = False
        self.assertEqual(DualGridContract().encode(raw)['local_grid'][4].sum(), 0)

    def test_vectorized_raster_matches_scalar_fractional_rectangles(self):
        from touhou_ai.dual_grid import paint_bullets
        rng = np.random.default_rng(19)
        bullets = [bullet(float(x), float(y), float(vx), float(vy)) |
                   {'hitbox_raw': [float(w), float(h)]}
                   for x, y, vx, vy, w, h in zip(rng.uniform(-200, 200, 300),
                       rng.uniform(190, 410, 300), rng.uniform(-8, 8, 300), rng.uniform(-8, 8, 300),
                       rng.uniform(.1, 30, 300), rng.uniform(.1, 30, 300))]
        origin = np.array([-96., 204.]); actual = np.zeros((6, 96, 96), np.float32)
        expected = np.zeros_like(actual); weights = np.zeros((96, 96), np.float32)
        for b in bullets:
            patch = rectangle(np.array(b['position']), np.array(b['hitbox_raw'])/2, origin, 2, (96, 96))
            if patch is None:
                continue
            sl, coverage = patch
            expected[1][sl] = np.maximum(expected[1][sl], coverage)
            weights[sl] += coverage
            for axis in range(2):
                expected[2+axis][sl] += coverage*(b['velocity_raw'][axis]/10)
        expected[2:4] /= np.maximum(weights, np.finfo(np.float32).tiny)
        paint_bullets(actual, bullets, origin)
        np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_policy_update_storage_roundtrip_and_save_reload(self):
        torch.set_num_threads(1); env = DualGridContract()
        model = PPO('MultiInputPolicy', env, n_steps=4, batch_size=2, n_epochs=1, seed=7,
                    rollout_buffer_class=GridRolloutBuffer,
                    policy_kwargs={'features_extractor_class': DualGridFeatures,
                                   'net_arch': {'pi': [256, 128], 'vf': [256, 128]}})
        raw = state(); raw['bullets'] = [bullet(12.3, 290.2)]; raw['enemies'] = [enemy(30, 40)]
        obs = env.encode(raw, {'damage': .1})
        buffer = GridRolloutBuffer(4, env.observation_space, env.action_space, device='cpu')
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(obs)
            action, value, logp = model.policy(tensor)
        for i in range(3):
            buffer.add(obs, action.cpu().numpy(), np.array([1.]), np.array([i == 0]), value, logp)
        finish_buffer(buffer, 3, value, True)
        sample = next(buffer.get(3))
        for k in ('local_grid', 'global_grid'):
            self.assertEqual(buffer.observations[k].dtype, np.float16)
            np.testing.assert_array_equal(sample.observations[k][0].float().numpy(), obs[k])
        values, log_probs, _ = model.policy.evaluate_actions(sample.observations, sample.actions.long())
        torch.testing.assert_close(log_probs, sample.old_log_prob)
        (values.square().mean()-log_probs.mean()).backward()
        for name in ('local', 'global_scene', 'player', 'reward'):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                                for p in getattr(model.policy.features_extractor, name).parameters()))
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(format_strings=[]))
        model.rollout_buffer = buffer
        model.train()  # Exercise actual SB3 PPO with half-precision stored observations.
        self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.zip'; model.save(path)
            loaded = PPO.load(path, env=env)
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
        self.assertEqual(model_metadata(model, CONTRACT, None)['bullet_scope'], SPEC)

    def test_old_checkpoint_cannot_silently_migrate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root/'artifacts/run'; run.mkdir(parents=True)
            checkpoint = run/'model.zip'; checkpoint.write_bytes(b'test')
            manifest = {'backend': 'real_th10', 'contract': 'th10-focused-bullets-v1', 'episodes': [
                {'checkpoint': checkpoint.name, 'reload_verified': True,
                 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]}
            with patch('touhou_ai.live_learning.ROOT', root):
                (run/'status.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'compatible'):
                    resume_manifest(checkpoint, extended=True, dual_grid=True)
                manifest['contract'] = CONTRACT
                (run/'status.json').write_text(json.dumps(manifest))
                self.assertEqual(resume_manifest(checkpoint, dual_grid=True)[0]['contract'], CONTRACT)


if __name__ == '__main__':
    unittest.main()

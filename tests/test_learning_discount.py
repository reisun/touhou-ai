import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.learning_discount import GAMMA, SPEC, verify_settings
from touhou_ai.live_learning import finish_buffer
from touhou_ai.timed_bomb import TimedBombPolicy, BombClock
from tests.test_timed_bomb import Tiny


class DiscountTests(unittest.TestCase):
    def test_config_real_model_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        settings = json.loads((Path(__file__).resolve().parents[1]/'configs/sharu-inspired-v1.json').read_text())['provisional_ppo']
        verify_settings(settings)
        self.assertEqual(settings['gae_lambda'], .95)
        self.assertEqual(settings['n_steps'], 2048)
        m = PPO(TimedBombPolicy, Tiny(), **settings)
        self.assertEqual(m.gamma, .9995)
        self.assertEqual(m.rollout_buffer.gamma, .9995)
        self.assertEqual(m.rollout_buffer.gae_lambda, .95)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'model.zip'; m.save(checkpoint)
            loaded = PPO.load(checkpoint, env=Tiny())
            self.assertEqual(loaded.gamma, .9995)
            self.assertEqual(loaded.rollout_buffer.gamma, .9995)
            self.assertEqual(loaded.gae_lambda, .95)
            self.assertEqual(loaded.n_steps, 2048)

    def test_returns_gae_terminal_truncation_and_episode_boundary(self):
        env = Tiny()
        for cls in (DictRolloutBuffer, GridRolloutBuffer):
            for terminal in (False, True):
                for starts in ([1, 0, 0], [1, 0, 1]):
                    with self.subTest(buffer=cls.__name__, terminal=terminal, starts=starts):
                        b = cls(8, env.observation_space, env.action_space, device='cpu', gamma=GAMMA, gae_lambda=.95)
                        rewards, values = [1., 2., 3.], [.25, .5, .75]
                        clock = BombClock()
                        for i in range(3):
                            obs = {'player': np.zeros(2, np.float32), 'bomb_clock': clock.encode()}
                            b.add(obs, np.zeros((1, 4)), np.array([rewards[i]]), np.array([starts[i]]),
                                  torch.tensor([values[i]]), torch.tensor([0.]))
                            clock.advance(2)
                        finish_buffer(b, 3, torch.tensor([5.]), terminal)
                        expected = np.zeros(3)
                        running = 0.
                        for i in reversed(range(3)):
                            alive = 1 - (int(terminal) if i == 2 else starts[i+1])
                            next_value = 5. if i == 2 else values[i+1]
                            delta = rewards[i] + GAMMA * next_value * alive - values[i]
                            running = delta + GAMMA * .95 * alive * running
                            expected[i] = running
                        np.testing.assert_allclose(b.advantages[:, 0], expected, rtol=2e-6)
                        np.testing.assert_allclose(b.returns[:, 0], expected + values, rtol=2e-6)
                        self.assertAlmostEqual(float(b.returns[-1, 0]), 3. if terminal else 3. + GAMMA*5, places=5)
                        self.assertEqual(b.buffer_size, 3)  # no rollout padding

    def test_discount_time_unit_and_old_contract_rejection(self):
        self.assertAlmostEqual(GAMMA**900, .637556398, places=7)
        self.assertAlmostEqual(GAMMA**1800, .406478159, places=7)
        self.assertEqual(SPEC['gameplay_frames_per_step'], 2)
        prior = {'discount_contract': SPEC, 'configured_ppo': {'gamma': GAMMA}}
        verify_settings({'gamma': GAMMA}, prior)
        for old in ({}, prior | {'configured_ppo': {'gamma': .997}},
                    prior | {'discount_contract': SPEC | {'gameplay_frames_per_step': 12}}):
            with self.assertRaisesRegex(ValueError, 'checkpoint discount'):
                verify_settings({'gamma': GAMMA}, old)
        with self.assertRaisesRegex(ValueError, 'configured gamma'):
            verify_settings({'gamma': .997})

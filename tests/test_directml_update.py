import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.directml_update import compatible_log_prob, update


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Dict({'player': gym.spaces.Box(-1, 1, (4,), dtype=np.float32)})
    action_space = gym.spaces.MultiDiscrete([3, 2])


class DirectMLRollbackTests(unittest.TestCase):
    def test_log_prob_and_gradient(self):
        logits = torch.randn(7, 9, requires_grad=True)
        actions = torch.arange(7)
        dist = torch.distributions.Categorical(logits=logits)
        expected = dist.log_prob(actions)
        actual = compatible_log_prob(dist, actions)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), logits)[0])

    def test_kl_early_stop_metadata(self):
        from types import SimpleNamespace
        from touhou_ai.directml_update import validate_update_metadata
        original = SimpleNamespace(num_timesteps=100, n_epochs=3, target_kl=.02, _n_updates=9)
        for completed in (1, 2, 3):
            candidate = SimpleNamespace(**vars(original))
            candidate._n_updates += completed
            self.assertEqual(validate_update_metadata(original, candidate), completed)
        for completed in (0, 4):
            candidate = SimpleNamespace(**vars(original)); candidate._n_updates += completed
            with self.assertRaises(ValueError): validate_update_metadata(original, candidate)
        original.target_kl = None
        candidate = SimpleNamespace(**vars(original)); candidate._n_updates += 1
        with self.assertRaises(ValueError): validate_update_metadata(original, candidate)

    def test_failed_worker_retries_untouched_parent(self):
        torch.set_num_threads(1)
        model = PPO('MultiInputPolicy', TinyEnv(), n_steps=8, batch_size=4, n_epochs=1, seed=7)
        model.set_logger(configure(None, []))
        buffer = GridRolloutBuffer(8, model.observation_space, model.action_space, device='cpu')
        buffer.full = True
        model.rollout_buffer = buffer
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        optimizer = model.policy.optimizer
        def cpu_retry():
            self.assertIs(model.policy.optimizer, optimizer)
            self.assertFalse(buffer.generator_ready)
            for k, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trial'
            with patch('touhou_ai.directml_update.subprocess.run', side_effect=RuntimeError('injected worker failure')):
                with patch.object(model, 'train', side_effect=cpu_retry) as train:
                    returned, info = update(model, buffer, directory)
            self.assertIs(returned, model)
            self.assertEqual(info['backend'], 'cpu_fallback')
            train.assert_called_once()
            self.assertFalse((directory / 'rollout.npz').exists())
            self.assertTrue((directory / 'before.zip').exists())


if __name__ == '__main__':
    unittest.main()

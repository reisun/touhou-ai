import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tests.test_live_learning import LiveLearningTests
from touhou_ai.live_learning import FocusedObservedContract
from touhou_ai.focused_policy import FocusedFeatures, reward_input


class FocusedPolicyTests(unittest.TestCase):
    def test_contract_and_history(self):
        env = FocusedObservedContract()
        raw = LiveLearningTests().state()
        raw['bullets'] = [{'position': [i, 390], 'velocity_raw': [1, 2],
                           'acceleration': [.1, .2]} for i in range(45, 0, -1)]
        current = env.encode(raw)
        self.assertNotIn('far_grid', current)
        self.assertEqual(sum(a.size for a in current.values()), 1610)
        np.testing.assert_array_equal(current['near_bullets'], current['bullets'][:10])
        np.testing.assert_array_equal(current['previous_rewards'], np.zeros(4))
        next_obs = env.encode(raw, {'hit': -10, 'damage': 1})
        self.assertEqual(next_obs['previous_rewards'][0], .5)
        self.assertEqual(next_obs['previous_rewards'][2], -.5)
        np.testing.assert_array_equal(current['previous_rewards'], np.zeros(4))
        np.testing.assert_array_equal(env.encode(raw)['previous_rewards'], np.zeros(4))
        self.assertTrue(env.observation_space.contains(next_obs))
        with self.assertRaises(ValueError):
            reward_input({'hit': float('nan')})

    def test_forward_backward_reload_and_individual_slots(self):
        torch.set_num_threads(1)
        env = FocusedObservedContract()
        model = PPO('MultiInputPolicy', env, n_steps=8, batch_size=4, seed=7,
                    policy_kwargs={'features_extractor_class': FocusedFeatures,
                                   'net_arch': {'pi': [256, 128], 'vf': [256, 128]}})
        raw = LiveLearningTests().state()
        raw['bullets'] = [{'position': [5, 390], 'velocity_raw': [1, 2]},
                          {'position': [20, 350], 'velocity_raw': [-1, 0]}]
        obs = env.encode(raw, {'damage': 1})
        tensor, _ = model.policy.obs_to_tensor(obs)
        features = model.policy.features_extractor(tensor)
        self.assertEqual(tuple(features.shape), (1, 256))
        changed = {k: v.clone() for k, v in tensor.items()}
        changed['near_bullets'][:, [0, 1]] = changed['near_bullets'][:, [1, 0]]
        self.assertFalse(torch.allclose(features, model.policy.features_extractor(changed)))
        action, value, logp = model.policy(tensor)
        loss = value.square().mean() - logp.mean()
        loss.backward()
        for name in ('near', 'reward', 'entities'):
            grads = [p.grad for p in getattr(model.policy.features_extractor, name).parameters()]
            self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
        model.policy.optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.zip'
            model.save(path)
            loaded = PPO.load(path, env=env)
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                          loaded.predict(obs, deterministic=True)[0])

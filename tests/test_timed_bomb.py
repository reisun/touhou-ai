import tempfile
import unittest
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from touhou_ai.timed_bomb import BombClock, TimedBombPolicy, INITIAL_PROBABILITY
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.live_learning import finish_buffer


class Tiny(gym.Env):
    observation_space = gym.spaces.Dict({'player': gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
                                       'bomb_clock': gym.spaces.Box(0, 1, (1,), dtype=np.float32)})
    action_space = gym.spaces.MultiDiscrete([9, 2, 2, 2])


def model():
    torch.set_num_threads(1)
    return PPO(TimedBombPolicy, Tiny(), n_steps=12, batch_size=6, n_epochs=1, seed=7,
               rollout_buffer_class=GridRolloutBuffer, policy_kwargs={'net_arch': [16]})


class TimedBombTests(unittest.TestCase):
    def test_single_pass_matches_actions_values_and_probabilities(self):
        from unittest.mock import patch
        policy = model().policy
        obs, _ = policy.obs_to_tensor({'player': np.zeros(2, dtype=np.float32),
                                     'bomb_clock': np.zeros(1, dtype=np.float32)})
        for clock in (0., .5):
            obs['bomb_clock'][:] = clock
            torch.manual_seed(123)
            expected = policy(obs)
            probs = [d.probs.clone() for d in policy.get_distribution(obs).distribution]
            torch.manual_seed(123)
            with patch.object(policy, '_latents', wraps=policy._latents) as latents:
                actual = policy.forward_with_distribution(obs)
                self.assertEqual(latents.call_count, 1)
            for a, b in zip(expected, actual[:3]):
                torch.testing.assert_close(a, b)
            for a, b in zip(probs, actual[3]):
                torch.testing.assert_close(a, b.probs)

    def test_live_collector_clock_and_transition_event_retention(self):
        import json
        import shutil
        from unittest.mock import Mock, patch
        from touhou_ai import live_learning as learner
        from tests.test_dual_grid import state as make_state
        from tests.test_live_rewards import damage
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'configs').mkdir(); (root/'.runtime').mkdir()
            shutil.copy(learner.ROOT/'configs/sharu-inspired-v1.json', root/'configs')
            (root/'.runtime/game.json').write_text(json.dumps({'Id': 123}))
            raw = make_state() | dict(stage_frame=2, pause_words=[2, 0], replay_mode=0, mode_flags=0)
            runtime = Mock(); runtime.snapshot.return_value = raw
            calls = []
            def step(mask, frames):
                calls.append(mask)
                return raw | dict(stage_frame=2+len(calls)*2, input_state_raw=[mask],
                                  player=raw['player'] | {'position': [2*(len(calls)%2), 300]},
                                  combat_reward_events=[damage(str(len(calls)), 0, 0, 1)])
            runtime.step_gameplay.side_effect = step
            # Simulate transition helper taking a new snapshot and dropping the batch.
            def bridge(rt, s, *args, **kwargs):
                return s | {'combat_reward_events': []}, {'automatic_frames': 0}
            with patch.object(learner, 'ROOT', root), patch.object(learner, 'LiveRuntime', return_value=runtime), \
                 patch.object(learner, 'game_command'), patch.object(learner, 'start_episode'), \
                 patch.object(learner, 'bridge_transition', side_effect=bridge), \
                 patch.object(learner, 'packet', return_value={'capabilities': {}, 'entities': {'bullets': [], 'items': []}, 'player': None}), \
                 patch.object(learner, 'publish_telemetry', return_value=True), \
                 patch('touhou_ai.model_monitor.model_metadata', return_value={}), \
                 patch('touhou_ai.live_acceptance.pause', return_value={'pause_words': [2, 2]}):
                report = learner.train(root/'artifacts/test', episodes=1, max_steps=32, dual_grid=True, detailed_logs=True)
            rows = [json.loads(x) for x in (root/'artifacts/test/episode-1.jsonl').read_text().splitlines()]
            self.assertTrue(rows, report)
            self.assertIsNone(rows[0]['previous_step_timing_ms'])
            timing = rows[1]['previous_step_timing_ms']
            self.assertEqual(timing['step'], 1)
            for key in ('inference_ms', 'step_observation_transition_ms', 'encode_ms', 'telemetry_ms', 'log_ms', 'publish_ms', 'total_ms'):
                self.assertGreaterEqual(timing[key], 0)
            self.assertGreaterEqual(timing['total_ms'], timing['inference_ms']+timing['encode_ms']+timing['log_ms'])
            self.assertEqual(json.loads((root/'artifacts/test/last-step-timing.json').read_text())['step'],32)
            self.assertEqual(report['status'], 'stopped')
            self.assertEqual(report['updates'], 0)
            self.assertEqual(report['configured_ppo']['gamma'], .9995)
            self.assertEqual(report['effective_ppo']['gamma'], .9995)
            self.assertEqual(report['discount_contract']['gameplay_frames_per_step'], 2)
            self.assertEqual(report['configured_ppo']['gae_lambda'], .95)
            self.assertEqual(report['configured_ppo']['n_steps'], 2048)
            self.assertEqual([i for i, r in enumerate(rows) if r['telemetry']['policy']['bomb_decision']], [0, 6, 12, 18, 24, 30])
            for i, row in enumerate(rows):
                if i % 6:
                    self.assertEqual(calls[i] & 2, 0)
                # v15: 1 HP of shot damage at Power 0 gives 15 / 1000.
                self.assertAlmostEqual(row['telemetry']['reward']['components']['damage'], .015 / 60)
                self.assertEqual(set(row['telemetry']['reward']['components']), {'damage', 'progress', 'hit', 'jitter'})
                self.assertAlmostEqual(row['telemetry']['reward']['total'],
                    .015 / 60 + row['telemetry']['reward']['components']['jitter'])
            self.assertEqual(sum(row['telemetry']['reward']['components']['jitter'] < 0 for row in rows), 5)

    def test_initial_probability_and_no_nondecision_draw_or_gradient(self):
        m = model()
        obs = {'player': torch.zeros(12, 2), 'bomb_clock': torch.tensor([[i % 6 / 6] for i in range(12)])}
        dist = m.policy.get_distribution(obs)
        torch.testing.assert_close(dist.distribution[3].probs[:, 1], torch.full((12,), INITIAL_PROBABILITY))
        self.assertAlmostEqual((1 - INITIAL_PROBABILITY)**25, .5)
        a, _, lp = m.policy(obs)
        self.assertTrue((a[obs['bomb_clock'][:, 0] != 0, 3] == 0).all())
        _, check, entropy = m.policy.evaluate_actions(obs, a)
        torch.testing.assert_close(lp, check)
        other_lp = sum(d.log_prob(a[:, i]) for i, d in enumerate(dist.distribution[:3]))
        other_entropy = sum(d.entropy() for d in dist.distribution[:3])
        mask = obs['bomb_clock'][:, 0] != 0
        torch.testing.assert_close(lp[mask], other_lp[mask])
        torch.testing.assert_close(entropy[mask], other_entropy[mask])
        m.policy.zero_grad(); (lp[mask].sum() + entropy[mask].sum()).backward()
        self.assertEqual(m.policy.action_net.weight.grad[-2:].abs().sum().item(), 0)
        self.assertEqual(m.policy.action_net.bias.grad[-2:].abs().sum().item(), 0)

    def test_clock_pulse_rollout_ppo_and_reload(self):
        m = model(); clock = BombClock()
        b = GridRolloutBuffer(12, m.observation_space, m.action_space, device='cpu')
        due = []
        for i in range(12):
            obs = {'player': np.zeros(2, np.float32), 'bomb_clock': clock.encode()}
            due.append(obs['bomb_clock'][0] == 0)
            with torch.no_grad():
                a, v, lp = m.policy(m.policy.obs_to_tensor(obs)[0])
            b.add(obs, a.numpy(), np.array([i % 3]), np.array([i == 0]), v, lp)
            clock.advance(2)
        self.assertEqual(np.flatnonzero(due).tolist(), [0, 6])
        finish_buffer(b, 12, v, True)
        m.rollout_buffer = b; m.set_logger(configure(None, []))
        bias = m.policy.action_net.bias.detach().clone()
        m.train()
        self.assertFalse(torch.equal(bias[-2:], m.policy.action_net.bias[-2:]))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'model.zip'; m.save(p); loaded = PPO.load(p)
            t = m.policy.obs_to_tensor(obs)[0]
            for x, y in zip(m.policy.evaluate_actions(t, a), loaded.policy.evaluate_actions(t, a)):
                torch.testing.assert_close(x, y)
            self.assertIsInstance(loaded.policy, TimedBombPolicy)

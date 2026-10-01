import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "learner dependencies required")
class LiveLearningTests(unittest.TestCase):
    def test_resume_requires_real_contract_and_checkpoint_integrity(self):
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from touhou_ai.live_learning import resume_manifest, CONTRACT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "artifacts/run"
            run.mkdir(parents=True)
            model = run / "model.zip"
            model.write_bytes(b"unit-test fixture, not a real checkpoint")
            manifest = {"backend": "real_th10", "contract": CONTRACT, "episodes": [{
                "checkpoint": "model.zip", "reload_verified": True,
                "checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}]}
            (run / "status.json").write_text(json.dumps(manifest))
            with patch('touhou_ai.live_learning.ROOT', root):
                self.assertEqual(resume_manifest(model)[0]["contract"], CONTRACT)
                model.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    resume_manifest(model)
                manifest["backend"] = "synthetic_rehearsal"
                (run / "status.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "compatible"):
                    resume_manifest(model)

    def test_viewer_file_contention_is_retried_without_affecting_learning(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from touhou_ai.live_learning import atomic_json, publish_telemetry
        original = Path.replace
        attempts = []
        def busy_once(path, target):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError("viewer has the old file open")
            return original(path, target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            with patch.object(Path, "replace", busy_once):
                atomic_json(path, {"ok": True})
            self.assertEqual(len(attempts), 2)
            self.assertIn('true', path.read_text())
        with patch('touhou_ai.telemetry_memory.publish', return_value=False):
            self.assertFalse(publish_telemetry({}))

    def state(self):
        return {"stage": 1, "stage_frame": 2, "lives_raw": 2, "power_raw": 0,
                "replay_mode": 0, "mode_flags": 0,
                "player": {"position": [0, 400], "velocity_raw": [0, 0], "status": 1,
                           "invincibility_raw": 118, "focus_raw": 0},
                "bullets": None, "enemies": [], "items": [], "lasers": []}

    def test_masked_missing_observation_and_disabled_bomb(self):
        from touhou_ai.live_learning import ObservedContract
        env = ObservedContract()
        obs = env.encode(self.state())
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs["player"][9], 0)
        self.assertEqual(env.encode(self.state() | {"bullets": []})["player"][9], 1)
        self.assertFalse(env.action_space.contains([0, 0, 0, 1]))
        with self.assertRaisesRegex(ValueError, "laser"):
            env.encode(self.state() | {"lasers": [{}]})

    def test_life_loss_only_no_disappearance_reward(self):
        from touhou_ai.live_learning import hit_events
        before = self.state()
        self.assertEqual(hit_events(before, before), [])
        events = hit_events(before, before | {"lives_raw": 1, "stage_frame": 4})
        self.assertEqual(events[0]["kind"], "hit")
        for update in ({"lives_raw": 0}, {"lives_raw": 4}, {"stage": 2}, {"replay_mode": 1}):
            with self.assertRaises(ValueError):
                hit_events(before, before | update)

    def test_extend_does_not_generate_hit_and_next_death_still_does(self):
        from touhou_ai.live_learning import hit_events
        from touhou_ai.live_rewards import LiveRewards, observed_events
        for stage in range(1, 7):
            before = self.state() | {'stage': stage}
            after = before | {'lives_raw': 3, 'stage_frame': 4}
            self.assertEqual(hit_events(before, after), [])
            rewards = LiveRewards()
            rewards.reset('extend')
            total, components = rewards.calculate('extend', observed_events(before, after))
            self.assertEqual(total, 0)
            self.assertEqual(components['hit'], 0)
            death = after | {'lives_raw': 2, 'stage_frame': 6}
            total, components = rewards.calculate('extend', observed_events(after, death))
            self.assertEqual(total, -1)
            self.assertEqual(components['hit'], -1)
        with self.assertRaises(ValueError):
            hit_events(before, after | {'mode_flags': 2})

    def test_partial_buffer_has_no_padding_and_bootstraps_truncation(self):
        import numpy as np
        import torch
        from stable_baselines3.common.buffers import DictRolloutBuffer
        from touhou_ai.live_learning import ObservedContract, finish_buffer
        env = ObservedContract()
        buffer = DictRolloutBuffer(8, env.observation_space, env.action_space, gamma=0.99)
        for i in range(3):
            buffer.add(env.encode(self.state()), np.zeros((1, 4)), np.array([0.0]),
                       np.array([i == 0]), torch.tensor([0.0]), torch.tensor([0.0]))
        finish_buffer(buffer, 3, torch.tensor([1.0]), False)
        self.assertEqual(buffer.buffer_size, 3)
        self.assertEqual(buffer.observations["player"].shape[0], 3)
        self.assertAlmostEqual(float(buffer.returns[-1, 0]), 0.99, places=5)

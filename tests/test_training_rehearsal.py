import importlib.util
from pathlib import Path
import tempfile
import unittest

from touhou_ai.training_readiness import readiness, require_live_ready


class ReadinessTests(unittest.TestCase):
    def test_live_launch_is_blocked(self):
        self.assertFalse(readiness()["ready_for_live_training"])
        self.assertTrue(readiness()["blockers"])
        with self.assertRaisesRegex(RuntimeError, "Live training blocked"):
            require_live_ready()


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "learner dependencies run in Docker")
class RehearsalTests(unittest.TestCase):
    def test_update_checkpoint_and_resume(self):
        from touhou_ai.training_rehearsal import rehearse
        profile = Path("/configs/sharu-inspired-v1.json")
        if not profile.exists():
            profile = Path(__file__).resolve().parents[1] / "configs/sharu-inspired-v1.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rehearsal"
            result = rehearse(profile, output, updates=1)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["synthetic_training_steps"], 64)
            self.assertEqual(result["gameplay_training_steps"], 0)
            self.assertTrue(result["parameters_changed"])
            self.assertTrue((output / "synthetic-resumed.zip").exists())
            with self.assertRaises(FileExistsError):
                rehearse(profile, output)

    def test_environment_reset_and_invalid_action(self):
        from touhou_ai.game_policy import load_profile
        from touhou_ai.training_rehearsal import RehearsalEnv
        profile = Path("/configs/sharu-inspired-v1.json")
        if not profile.exists():
            profile = Path(__file__).resolve().parents[1] / "configs/sharu-inspired-v1.json"
        env = RehearsalEnv(load_profile(profile), horizon=2)
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0, 0])
        observation, _ = env.reset()
        self.assertTrue(env.observation_space.contains(observation))
        with self.assertRaises(ValueError):
            env.step([9, 0, 0, 0])
        self.assertEqual(env.tick, 0)
        _, reward, _, _, info = env.step([0, 1, 0, 0])
        self.assertGreater(reward, 0)
        self.assertEqual(info["source"], "synthetic_rehearsal")
        self.assertTrue(env.step([0, 0, 0, 0])[3])
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0, 0])

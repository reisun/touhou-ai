import importlib.util
from pathlib import Path
import tempfile
import unittest

from touhou_ai.game_rewards import EventRewards


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.rewards = EventRewards()
        self.rewards.reset("episode")

    def calculate(self, events):
        return self.rewards.calculate("episode", events)[0]

    def event(self, kind, **kwargs):
        return {"id": kind, "kind": kind, "confirmed": True, **kwargs}

    def test_sparse_rewards_and_duplicate_protection(self):
        self.assertEqual(self.calculate([]), 0)
        event = self.event("stage_clear", subject="stage1", remaining_lives=2)
        self.assertEqual(self.calculate([event]), 30)
        self.assertEqual(self.calculate([event]), 0)
        self.assertEqual(self.calculate([event | {"id": "different"}]), 0)
        with self.assertRaises(ValueError):
            self.calculate([event | {"remaining_lives": 3}])

    def test_bomb_damage_exploit_and_fixed_hit_penalty(self):
        event = self.event("damage", amount=50, phase_initial_hp=100, bomb_active=True)
        self.assertEqual(self.calculate([event]), 0)
        self.assertEqual(self.calculate([event | {"id": "shot", "bomb_active": False}]), 0.5)
        self.assertEqual(self.calculate([self.event("bomb")]), -0.5)
        self.assertEqual(self.calculate([self.event("hit", frame=1)]), -5)
        self.assertEqual(self.calculate([self.event("hit", id="late", frame=90000)]), -5)

    def test_invalid_batch_is_atomic_and_reset_scopes_ids(self):
        event = self.event("kill", subject="enemy-lifetime-1")
        with self.assertRaises(ValueError):
            self.calculate([event, self.event("power_loss")])
        self.assertEqual(self.calculate([event]), 0.5)
        with self.assertRaises(ValueError):
            self.calculate([self.event("bomb", confirmed=False)])
        self.rewards.reset("next")
        self.assertEqual(self.rewards.calculate("next", [event])[0], 0.5)
        with self.assertRaises(ValueError):
            self.calculate([])


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "learner dependencies run in Docker")
class PolicyTests(unittest.TestCase):
    def setUp(self):
        from touhou_ai.game_policy import NumericalEncoder, load_profile
        from touhou_ai.policy_check import synthetic_state
        self.path = Path("/configs/sharu-inspired-v1.json")
        if not self.path.exists():
            self.path = Path(__file__).resolve().parents[1] / "configs/sharu-inspired-v1.json"
        self.encoder = NumericalEncoder(load_profile(self.path))
        self.state = synthetic_state()

    def test_empty_shapes_bounds_and_actions(self):
        from touhou_ai.game_policy import decode_action
        observation = self.encoder.encode(self.state)
        self.assertTrue(self.encoder.space.contains(observation))
        self.assertEqual(observation["bullets"].sum(), 0)
        self.assertAlmostEqual(float(observation["player"][8]), 384/448)
        self.assertEqual(decode_action([2, 1, 1, 0]), ["up", "right", "z", "shift"])
        with self.assertRaises(ValueError):
            decode_action([9, 0, 0, 0])
        with self.assertRaises(ValueError):
            decode_action([1.5, 0, 0, 0])
        self.state["player"]["y"] = 0
        with self.assertRaises(ValueError):
            self.encoder.encode(self.state)

    def test_cpa_far_field_and_order_independence(self):
        import numpy as np
        self.state["bullets"] = [{"id": str(i), "x": 192, "y": 300-i,
                                  "vx": 0, "vy": 100, "ax": 0, "ay": 0, "radius_x": 3, "radius_y": 3}
                                 for i in range(129)]
        first = self.encoder.encode(self.state)
        self.assertAlmostEqual(float(first["bullets"][0, -1]), 0.5)
        self.assertAlmostEqual(float(first["bullets"][0, -2]), 0)
        self.assertEqual(np.count_nonzero(first["far_grid"][0]), 1)
        self.state["bullets"].reverse()
        second = self.encoder.encode(self.state)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.state["bullets"][0]["ax"] = float("nan")
        with self.assertRaises(ValueError):
            self.encoder.encode(self.state)

    def test_policy_roundtrip_and_live_learning_refusal(self):
        from touhou_ai.game_policy import load_profile
        from touhou_ai.policy_check import ContractOnlyEnv, check
        with tempfile.TemporaryDirectory() as directory:
            result = check(self.path, Path(directory) / "policy")
            self.assertEqual(result["gameplay_training_steps"], 0)
            import json
            diagnostic = json.loads((Path(directory) / "policy/policy-telemetry.json").read_text())
            self.assertEqual(len(diagnostic["directions"]), 9)
            self.assertAlmostEqual(sum(diagnostic["directions"]), 1, places=5)
            self.assertFalse(diagnostic["trained"])
            for key in ("shoot", "focus", "bomb"):
                self.assertTrue(0 <= diagnostic[key] <= 1)
        with self.assertRaises(RuntimeError):
            ContractOnlyEnv(load_profile(self.path)).reset()

import unittest
import numpy as np
from tests import test_live_learning as fixtures
from touhou_ai.live_learning import FocusedObservedContract
from touhou_ai.telemetry import packet

class BulletScopeTests(unittest.TestCase):
    def test_full_scene_selection_and_grid_match_policy(self):
        raw = fixtures.LiveLearningTests().state()
        near = {'position': [0, 390], 'velocity_raw': [1, 2]}
        far = {'position': [0, 0], 'velocity_raw': [8, -9]}
        raw['bullets'] = [far]*10+[near]*129
        raw['items'] = [far]*42
        display = packet(raw, 'test')['ai_observation']
        self.assertEqual(len(display['bullets']), 40)
        self.assertEqual(len(display['items']), 40)
        self.assertEqual(display['bullet_grid'], [])
        for env in (FocusedObservedContract(),):
            obs = env.encode(raw)
            self.assertEqual(obs['items'][:, 0].sum(), 40)
            self.assertEqual(env.encode(raw | {'bullets': [far]})['bullets'][0, 0], 1)
            self.assertNotIn('far_grid', obs)
            for i, entity in enumerate(display['bullets']):
                np.testing.assert_allclose(obs['bullets'][i, 1:3],
                    [(entity['position'][0]-192)/384, (entity['position'][1]-400)/448], atol=1e-6)

    def test_missing_data(self):
        raw = fixtures.LiveLearningTests().state()
        self.assertIsNone(packet(raw, 'test')['ai_observation']['bullets'])
        raw['player'] = None
        self.assertIsNone(packet(raw, 'test')['ai_observation']['items'])

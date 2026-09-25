import unittest

from touhou_ai.laser_acceptance import assess
from touhou_ai.live_features import laser_collision


class LaserAcceptanceTests(unittest.TestCase):
    def test_recorded_real_hit_at_frame_7895(self):
        row = {'state': 2, 'position': [29.665096282958984, 209.37100219726562],
               'raw': [1.435217261314392, 180, 14],
               'origin': [32.09804916381836, 227.20582580566406],
               'angle': 1.435217261314392, 'length': 144, 'width': 7,
               'player': [47.13999938964844, 369.0099792480469], 'halfbox': [1, 1],
               'player_status': 1, 'invincibility': 0, 'dialogue': 0, 'result': 1}
        result = assess(row)
        self.assertTrue(result['expected_hit'])
        self.assertFalse(result['mismatch'])
        self.assertLess(result['parameter_error'], .00001)
        self.assertAlmostEqual(result['margin'], -.237085, places=5)

    def test_live_validation_scope_is_not_generalized(self):
        self.assertTrue(laser_collision('line', 2, [0, 0], 0, 180, 14)['field_validated'])
        for kind, state, length, width in [('infinite', 2, 180, 14), ('line', 1, 180, 14),
                                          ('line', 2, 181, 14), ('line', 2, 180, 16)]:
            self.assertFalse(laser_collision(kind, state, [0, 0], 0, length, width)['field_validated'])
        self.assertFalse(laser_collision('line', 2, [0, 0], 0, 180, 14)['activation_transition_validated'])

    def test_boundary_and_immunity(self):
        row = {'state': 2, 'position': [0, 0], 'raw': [0, 100, 14],
               'origin': [10, 0], 'angle': 0, 'length': 80, 'width': 7,
               'player': [50, 4.4], 'halfbox': [1, 1], 'player_status': 1,
               'invincibility': 0, 'dialogue': 0, 'result': 1}
        self.assertFalse(assess(row)['mismatch'])
        self.assertAlmostEqual(assess(row)['margin'], -.1)
        self.assertEqual(assess(row)['parameter_error'], 0)
        outside = row | {'player': [50, 4.6], 'result': 0}
        self.assertFalse(assess(outside)['mismatch'])
        self.assertAlmostEqual(assess(outside)['margin'], .1)
        self.assertTrue(assess(outside | {'result': 1})['mismatch'])
        for field, value in [('invincibility', 1), ('dialogue', 1), ('player_status', 4)]:
            result = assess(row | {field: value, 'result': 0})
            self.assertFalse(result['eligible'])
            self.assertFalse(result['mismatch'])
        self.assertFalse(assess(row | {'player': [91.1, 0], 'result': 0})['mismatch'])

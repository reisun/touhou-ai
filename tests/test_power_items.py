import unittest
import numpy as np
from test_dual_grid import state
from touhou_ai.dual_grid import DualGridContract, GLOBAL_CHANNELS
from touhou_ai.power_items import MAX_ITEMS


class PowerItemTests(unittest.TestCase):
    def test_types_separate_sum_and_preserve_linear_ratio(self):
        raw = state()
        raw['items'] = [dict(type=t, position=[x, 100]) for t, x in
                        [(1, 0), (10, 0), (4, 8), (11, 8), (2, 0), (9, 0)]]
        grid = DualGridContract().encode(raw)['global_grid']
        p = grid[GLOBAL_CHANNELS.index('power_amount')]
        self.assertAlmostEqual(float(p[12, 24]), .02, places=4)
        self.assertAlmostEqual(float(p[12, 25]), .4, places=3)
        self.assertEqual(np.count_nonzero(grid[6]), 1)
        self.assertAlmostEqual(float(grid[6, 12, 24]), np.log(3)/np.log(17), places=3)

    def test_single_small_power_survives_policy_storage(self):
        for kind in (1, 10):
            raw = state()
            raw['items'] = [dict(type=kind, position=[0, 100])]
            obs = DualGridContract().encode(raw)
            layer = obs['global_grid'][GLOBAL_CHANNELS.index('power_amount')]
            self.assertEqual(float(layer[12, 24]), float(np.float16(.01)))
            self.assertEqual(np.count_nonzero(layer), 1)
            self.assertEqual(obs['global_grid'][GLOBAL_CHANNELS.index('item_density')].sum(), 0)

    def test_power_sum_above_one_does_not_saturate(self):
        raw = state(); env = DualGridContract()
        for count in (6, MAX_ITEMS):
            raw['items'] = [dict(type=4, position=[0, 100]) for _ in range(count)]
            obs = env.encode(raw)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(obs['global_grid'][11, 12, 24], np.float16(count*.2))
            self.assertEqual(obs['global_grid'][6].sum(), 0)

    def test_boundaries_and_unknown_types(self):
        raw = state(); env = DualGridContract()
        raw['items'] = [dict(type=1, position=[x, y]) for x, y in
                        [(-193, 0), (-192, 0), (192, 100), (0, 448), (191, 447)]]
        self.assertEqual(np.count_nonzero(env.encode(raw)['global_grid'][11]), 2)
        for kind in (None, True, 1.0, 0, 12):
            with self.assertRaisesRegex(ValueError, 'item type'):
                env.encode(raw | {'items': [dict(type=kind, position=[0, 100])]})

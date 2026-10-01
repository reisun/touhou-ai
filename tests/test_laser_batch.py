"""Compare batching with the original continuous scalar SAT implementation."""
import unittest
import numpy as np
from touhou_ai.live_action_grid import live_risk_grid, swept_laser_contact
from touhou_ai.spatial_input_candidates import CandidateEnv, risk_grid, TILES, LOW, HIGH
from touhou_ai.fixed_dodge import DIRECTIONS


def reference(raw):
    out = risk_grid(raw | {'lasers': []})
    lasers = [x['collision'] for x in raw['lasers'] if x['collision']['active']]
    start = np.asarray(raw['player']['position'], float)
    half = np.asarray(raw['player']['hitbox_raw'], float)
    for focus, speed in enumerate((4.5, 2.)):
        for direction, (iy, ix) in enumerate(TILES):
            velocity = DIRECTIONS[direction] * speed
            times = [0., 2.]
            for axis in (0, 1):
                if velocity[axis]:
                    t = ((HIGH if velocity[axis] > 0 else LOW)[axis] - start[axis]) / velocity[axis]
                    if 0 < t < 2:
                        times.append(t)
            times.sort()
            for a, b in zip(times[:-1], times[1:]):
                p = np.clip(start + a * velocity, LOW, HIGH)
                q = np.clip(start + b * velocity, LOW, HIGH)
                if any(swept_laser_contact(p, q, half, c) for c in lasers):
                    out[focus, iy, ix] = out[focus + 2, iy, ix] = 1
                    break
    return out


class LaserBatchTests(unittest.TestCase):
    def test_random_and_wall_parity(self):
        rng = np.random.default_rng(991)
        env = CandidateEnv('action_grid'); env.reset(seed=7)
        raw = env.raw_observation(); raw['bullets'] = []
        for i in range(250):
            pos = rng.uniform(LOW, HIGH)
            if i % 3 == 0:
                pos = np.where(rng.integers(2, size=2), HIGH, LOW) + rng.uniform(-3, 3, size=2)
                pos = np.clip(pos, LOW, HIGH)
            raw['player']['position'] = pos.tolist()
            raw['lasers'] = [{'collision': dict(origin=(pos + rng.uniform(-30, 30, size=2)).tolist(),
                angle=rng.uniform(-np.pi, np.pi), length=rng.uniform(0, 80),
                width=rng.uniform(0, 8), active=bool(rng.integers(2)))} for _ in range(12)]
            np.testing.assert_array_equal(live_risk_grid(raw), reference(raw))

    def test_tangent_and_stationary_parity(self):
        env = CandidateEnv('action_grid'); env.reset(seed=7)
        raw = env.raw_observation(); raw['bullets'] = []
        pos = np.asarray(raw['player']['position'])
        half = np.asarray(raw['player']['hitbox_raw'])
        for epsilon in (0., -1e-12, 1e-12):
            for angle in (0., np.pi / 2, np.pi, np.pi / 4):
                raw['lasers'] = [{'collision': dict(origin=(pos + [half[0] + 1 + epsilon, 0]).tolist(),
                    angle=angle, length=20., width=2., active=True)}]
                np.testing.assert_array_equal(live_risk_grid(raw), reference(raw))


if __name__ == '__main__':
    unittest.main()

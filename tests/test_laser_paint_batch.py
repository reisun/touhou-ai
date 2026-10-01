"""Frozen pre-optimization rasterizer and edge/overlap parity checks."""
import math
import unittest
import numpy as np
from touhou_ai.dual_grid import paint_laser, pair


def reference(grid, channel, verified_channel, collision, origin, cell):
    if not collision['active']:
        return
    pos = pair(collision['origin'], 'laser origin')
    angle, length, width = [float(collision[k]) for k in ('angle', 'length', 'width')]
    if not all(math.isfinite(v) for v in (angle, length, width)) or min(length, width) < 0:
        raise ValueError('invalid laser geometry')
    c, s = math.cos(angle), math.sin(angle)
    direction, normal = np.array([c, s]), np.array([-s, c])
    corners = np.array([pos+direction*t+normal*w for t in (0, length) for w in (-width/2, width/2)])
    lo = np.maximum(np.floor((corners.min(0)-origin)/cell), 0).astype(int)
    hi = np.minimum(np.ceil((corners.max(0)-origin)/cell), [grid.shape[2], grid.shape[1]]).astype(int)
    x0, y0 = lo; x1, y1 = hi
    if x0 >= x1 or y0 >= y1:
        return
    coverage = np.zeros((y1-y0, x1-x0), dtype=np.float32)
    for ox in (.25, .75):
        for oy in (.25, .75):
            dx = origin[0]+(np.arange(x0, x1)[None, :]+ox)*cell-pos[0]
            dy = origin[1]+(np.arange(y0, y1)[:, None]+oy)*cell-pos[1]
            along, across = dx*c+dy*s, -dx*s+dy*c
            coverage += ((along >= 0) & (along <= length) & (np.abs(across) <= width/2))*.25
    sl = (slice(y0, y1), slice(x0, x1))
    grid[channel][sl] = np.maximum(grid[channel][sl], coverage)
    if collision['field_validated']:
        grid[verified_channel][sl] = np.maximum(grid[verified_channel][sl], coverage)


class LaserPaintBatchTests(unittest.TestCase):
    def test_random_overlap_viewport_and_boundary(self):
        rng = np.random.default_rng(993)
        for shape, cell in (((2, 96, 96), 2), ((2, 56, 48), 8)):
            for scene in range(80):
                origin = rng.uniform(-300, 300, 2)
                old = np.zeros(shape, np.float32); new = old.copy()
                for j in range(12):
                    collision = dict(active=j % 7 != 0, field_validated=j % 3 == 0,
                        origin=(origin + rng.uniform(-100, 400, 2)).tolist(),
                        angle=rng.uniform(-math.pi, math.pi), length=rng.uniform(0, 800), width=rng.uniform(0, 100))
                    if scene < 8:
                        collision.update(origin=(origin + [.25*cell, .75*cell]).tolist(),
                                         angle=scene*math.pi/4, width=(0, cell, 2*cell)[j%3])
                    reference(old, 0, 1, collision, origin, cell)
                    paint_laser(new, 0, 1, collision, origin, cell)
                np.testing.assert_array_equal(new, old)

    def test_invalid_geometry(self):
        c = dict(active=True, field_validated=False, origin=[0, 0], angle=0, length=10, width=2)
        for key, value in (('width', -1), ('length', float('nan')), ('angle', float('inf')), ('origin', [0])):
            for fn in (reference, paint_laser):
                with self.assertRaises(ValueError):
                    fn(np.zeros((2, 96, 96), np.float32), 0, 1, c | {key: value}, np.zeros(2), 2)


if __name__ == '__main__':
    unittest.main()

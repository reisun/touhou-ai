"""Causal vibration measurement and time-scaled reward; no new grid work."""
from collections import deque
from math import hypot, isfinite
from touhou_ai.motion_jitter import MotionJitter

SOURCE = 'actual_displacement_vibration_36f_v1'
SPEC = dict(version=SOURCE, window_frames=36, ema_alpha=.25,
            reference_pixels_per_decision=4., decisions_per_second=30,
            maximum_penalty_per_second=.1,
            charge_on='current_nonzero_movement_change', minimum_window_decisions=18)

def vibration(moves):
    sx, sy = moves[0]
    residual = alternating = 0.
    previous = last_acceleration = None
    for x, y in moves:
        sx += .25*(x-sx); sy += .25*(y-sy)
        residual += hypot(x-sx, y-sy)
        if previous is not None:
            a = (x-previous[0], y-previous[1])
            size = hypot(*a)
            if size > .01:
                if last_acceleration is not None:
                    b = last_acceleration
                    alternating += max(0., -(a[0]*b[0]+a[1]*b[1])/(size*hypot(*b)))
                last_acceleration = a
        previous = (x, y)
    return residual/len(moves)*min(1., max(0., (alternating-1.)/3.))

class VibrationReward(MotionJitter):
    # Inherit validated death/respawn/frame-gap exclusions; add returns a fraction.
    def reset(self):
        self.moves = deque(maxlen=18)
        self.score = 0.

    def add(self, dx, dy):
        if not all(isfinite(v) for v in (dx, dy)):
            self.reset()
            return 0.
        previous = self.moves[-1] if self.moves else None
        self.moves.append((dx, dy))
        self.score = vibration(self.moves) if len(self.moves)==18 else 0.
        # Old residuals must not charge a stop or a newly steady direction.
        if (previous is None or hypot(dx,dy)<=.01
                or hypot(dx-previous[0],dy-previous[1])<=.01):
            return 0.
        return min(1., self.score/4.)/30.

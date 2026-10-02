"""Causal vibration measurement and time-scaled reward; no new grid work."""
from collections import deque
from math import hypot, isfinite
from touhou_ai.motion_jitter import MotionJitter

SOURCE = 'actual_displacement_averaged_vibration_36f_v2'
SPEC = dict(version=SOURCE, window_frames=36, velocity_average_frames=24,
            ema_alpha=.25, position_reference='least_squares_constant_velocity',
            velocity_residual_horizon_decisions=3.,
            reference_position_pixels=4., decisions_per_second=30,
            maximum_penalty_per_second=.1,
            charge_on='current_nonzero_movement_change', minimum_window_decisions=18)


def vibration_components(moves):
    """Use past samples only: averaged velocity and position jitter about its trend."""
    moves = list(moves)
    positions = [(0., 0.)]
    alternating = 0.
    previous = last_acceleration = None
    for x, y in moves:
        px, py = positions[-1]
        positions.append((px+x, py+y))
        if previous is not None:
            a = (x-previous[0], y-previous[1])
            size = hypot(*a)
            if size > .01:
                if last_acceleration is not None:
                    b = last_acceleration
                    alternating += max(0., -(a[0]*b[0]+a[1]*b[1])/(size*hypot(*b)))
                last_acceleration = a
        previous = (x, y)
    # The best constant-velocity path fits drift, including a regular tap cadence.
    count = len(positions)
    center = (count-1)/2.
    mean_x = sum(p[0] for p in positions)/count
    mean_y = sum(p[1] for p in positions)/count
    denominator = sum((i-center)**2 for i in range(count))
    vx = sum((i-center)*p[0] for i,p in enumerate(positions))/denominator
    vy = sum((i-center)*p[1] for i,p in enumerate(positions))/denominator
    position_rms = (sum((x-mean_x-vx*(i-center))**2 +
                        (y-mean_y-vy*(i-center))**2
                        for i,(x,y) in enumerate(positions))/count)**.5
    # 12 decisions = 24F, averaging common 2/3/4/6-decision tap cadences.
    averaged = [(sum(x for x,y in moves[i:i+12])/12.,
                 sum(y for x,y in moves[i:i+12])/12.)
                for i in range(len(moves)-11)]
    residual = 0.
    if averaged:
        sx, sy = averaged[0]
        for x, y in averaged:
            sx += .25*(x-sx); sy += .25*(y-sy)
            residual += hypot(x-sx,y-sy)
        residual /= len(averaged)
    repeat_weight = min(1., max(0., (alternating-1.)/3.))
    # Convert speed residual to pixels over 3 decisions before combining units.
    score = hypot(position_rms, 3.*residual)*repeat_weight
    return dict(position_rms_pixels=position_rms,
                averaged_velocity_residual=residual,
                repeat_weight=repeat_weight, vibration=score)


def vibration(moves):
    return vibration_components(moves)['vibration']


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

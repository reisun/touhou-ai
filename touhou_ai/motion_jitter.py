"""Bounded, nearly collinear backtracking; charge only a new reversal."""
from collections import deque
from math import hypot, isfinite


SOURCE = 'actual_displacement_backtrack_36f_v2'
SPEC = dict(version=SOURCE, window_frames=36, alignment_cosine=.95,
            cancellation_fraction=.75, minimum_reversals=2, cooldown_frames=12,
            charge_on='current_nonzero_reversal', consume_history=True)


class MotionJitter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.moves = deque(maxlen=18)
        self.cooldown = 0

    def add(self, dx, dy):
        if not all(isfinite(v) for v in (dx, dy)):
            self.reset()
            return False
        self.moves.append((dx, dy))
        self.cooldown = max(0, self.cooldown - 1)
        norm = hypot(dx, dy)
        if norm <= .01 or self.cooldown:
            return False
        # Pauses consume time in the window but do not erase direction.
        moving = [v for v in self.moves if hypot(*v) > .01]
        if len(moving) < 3:
            return False
        last = moving[-2]
        if dx*last[0]+dy*last[1] > -.95*norm*hypot(*last):
            return False
        # Examine only the contiguous nearly-collinear suffix ending now.
        # A triangle or a perpendicular turn breaks the same-path evidence.
        axis = (dx/norm, dy/norm)
        length = sx = sy = 0.
        flips = 0
        last_sign = None
        for x, y in reversed(moving):
            size = hypot(x, y)
            projection = x*axis[0]+y*axis[1]
            if abs(projection) < .95*size:
                break
            sign = 1 if projection > 0 else -1
            flips += last_sign is not None and sign != last_sign
            last_sign = sign
            length += size
            sx += x
            sy += y
            if flips >= 2 and length >= 1 and hypot(sx, sy) <= .25*length:
                self.cooldown = 6
                self.moves.clear()  # Never charge the same old excursion again.
                return True
        return False

    def observe(self, before, after):
        """Exclude death/respawn, menus, stage changes and missing 2F samples."""
        players = [s.get('player') for s in (before, after)]
        if (not all(p and p.get('status') == 1 for p in players)
                or before['stage'] != after['stage']
                or after['stage_frame']-before['stage_frame'] != 2
                or after['lives_raw'] < before['lives_raw']
                or any(s.get('mode_flags', 0) not in (0, 4) for s in (before, after))):
            self.reset()
            return False
        dx, dy = (b-a for a, b in zip(players[0]['position'], players[1]['position']))
        if not all(isfinite(v) for v in (dx, dy)) or hypot(dx, dy) > 16:
            self.reset()
            return False
        return self.add(dx, dy)

"""Six actual 2F displacements; same detector as the jitter penalty experiment."""
from collections import deque
from math import hypot, isfinite


class MotionJitter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.moves = deque(maxlen=6)
        self.cooldown = 0

    def add(self, dx, dy):
        self.moves.append((dx, dy))
        self.cooldown = max(0, self.cooldown - 1)
        if len(self.moves) < 6 or self.cooldown:
            return False
        norms = [hypot(*v) for v in self.moves]
        length = sum(norms)
        reversals = sum(
            a[0]*b[0]+a[1]*b[1] < -.5*na*nb and na > .01 and nb > .01
            for a, b, na, nb in zip(self.moves, list(self.moves)[1:], norms, norms[1:]))
        net = hypot(sum(v[0] for v in self.moves), sum(v[1] for v in self.moves))
        flagged = length >= 1 and 1-net/length >= .75 and reversals >= 2
        if flagged:
            self.cooldown = 6
        return flagged

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

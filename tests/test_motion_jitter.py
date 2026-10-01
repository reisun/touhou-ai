import unittest
from touhou_ai.motion_jitter import MotionJitter
from touhou_ai.live_rewards import LiveRewards


class MotionJitterTests(unittest.TestCase):
    def test_matches_simulator_formula(self):
        import numpy as np
        rng = np.random.default_rng(7)
        detector = MotionJitter()
        moves, cooldown = [], 0
        for delta in rng.integers(-4, 5, size=(10000, 2)):
            moves = (moves+[delta])[-6:]
            cooldown = max(0, cooldown-1)
            flagged = False
            if len(moves) == 6 and not cooldown:
                v = np.array(moves); norm = np.linalg.norm(v, axis=1); length = norm.sum()
                rev = (np.sum(v[1:]*v[:-1],axis=1) < -.5*norm[1:]*norm[:-1]) & (norm[1:]>.01) & (norm[:-1]>.01)
                flagged = bool(length >= 1 and 1-np.linalg.norm(v.sum(0))/length >= .75 and rev.sum() >= 2)
                if flagged: cooldown = 6
            self.assertEqual(detector.add(*delta), flagged)

    def test_reversal_and_cooldown(self):
        detector = MotionJitter()
        flags = [i for i in range(24) if detector.add(2 if i % 2 else -2, 0)]
        self.assertEqual(flags, [5, 11, 17, 23])

    def test_straight_stationary_and_single_turn(self):
        for path in ([(2, 0)]*24, [(0, 0)]*24, [(2, 0)]*3+[(-2, 0)]*3):
            detector = MotionJitter()
            self.assertFalse(any(detector.add(*v) for v in path))

    def test_live_discontinuities_clear_history(self):
        def state(frame, x=0):
            return dict(stage=1, stage_frame=frame, lives_raw=2,
                        player=dict(status=1, position=[x, 300]))
        for change in (dict(stage=2), dict(stage_frame=4), dict(lives_raw=1),
                       dict(player=None), dict(player=dict(status=2, position=[0,300])),
                       dict(player=dict(status=1, position=[100,300]))):
            detector = MotionJitter()
            for i in range(5): detector.add(2 if i%2 else -2, 0)
            self.assertFalse(detector.observe(state(0), state(2) | change))
            self.assertEqual(len(detector.moves), 0)

    def test_live_event_and_scalar_reward_dedup(self):
        detector = MotionJitter()
        flagged = []
        for i in range(6):
            before = dict(stage=1, stage_frame=i*2, lives_raw=2,
                          player=dict(status=1, position=[(i%2)*2, 300]))
            after = before | dict(stage_frame=i*2+2,
                player=dict(status=1, position=[((i+1)%2)*2, 300]))
            flagged.append(detector.observe(before, after))
        self.assertEqual(flagged, [False]*5+[True])
        rewards = LiveRewards(); rewards.reset('test')
        event = dict(id='jitter:1:12', kind='jitter', confirmed=True,
                     source='actual_displacement_12f_v1')
        total, parts = rewards.calculate('test', [event])
        self.assertEqual(total, -.1)
        self.assertEqual(parts['jitter'], -.1)
        self.assertEqual(parts['hit'], 0)
        self.assertEqual(rewards.calculate('test', [event])[0], 0)

    def test_previous_reward_input_shape_is_preserved(self):
        import numpy as np
        from touhou_ai.focused_policy import reward_input
        np.testing.assert_array_equal(reward_input({'hit': -1, 'jitter': -.1}),
                                      reward_input({'hit': -1}))

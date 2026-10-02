import unittest
from touhou_ai.motion_jitter import MotionJitter
from touhou_ai.live_rewards import LiveRewards


class MotionJitterTests(unittest.TestCase):
    def test_backtracking_with_pauses_and_longer_legs(self):
        for path in ([(-4, 0), (0, 0), (4, 0), (0, 0)]*8,
                     ([(-4, 0)]*5+[(4, 0)]*5)*4):
            d = MotionJitter()
            flags = [i for i, v in enumerate(path) if d.add(*v)]
            self.assertTrue(flags)
            self.assertTrue(all(b-a >= 6 for a, b in zip(flags, flags[1:])))
            self.assertTrue(all(path[i] != (0, 0) for i in flags))

    def test_triangle_and_forward_zigzag_are_not_backtracking(self):
        for path in ([(0, 0)]*3+[(6.36, 6.36), (-9, 0), (2.82, -2.82)],
                     [(4, 4), (-4, 4)]*20):
            d = MotionJitter()
            self.assertFalse(any(d.add(*v) for v in path))

    def test_never_charge_stop_or_stale_history(self):
        d = MotionJitter()
        for x in [-4, -4, -4, 4, -4, 4]:
            d.add(x, 0)
        self.assertFalse(d.add(0, 0))
        for _ in range(18):
            self.assertFalse(d.add(0, 0))
        self.assertFalse(d.add(-4, 0))

    def test_repeated_same_direction_never_charged(self):
        d = MotionJitter()
        for x in [-4, -4, 4, -4, 4]:
            d.add(x, 0)
        for _ in range(30):
            self.assertFalse(d.add(4, 0))

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
        self.assertEqual(flagged, [False, False, False, True, False, False])
        rewards = LiveRewards(); rewards.reset('test')
        event = dict(id='jitter:1:12', kind='jitter', confirmed=True,
                     source='actual_displacement_averaged_vibration_36f_v2', amount=1./30)
        total, parts = rewards.calculate('test', [event])
        self.assertAlmostEqual(total, -.1/30)
        self.assertAlmostEqual(parts['jitter'], -.1/30)
        self.assertEqual(parts['hit'], 0)
        self.assertEqual(rewards.calculate('test', [event])[0], 0)

    def test_previous_reward_input_shape_is_preserved(self):
        import numpy as np
        from touhou_ai.focused_policy import reward_input
        np.testing.assert_array_equal(reward_input({'hit': -1, 'jitter': -.1}),
                                      reward_input({'hit': -1}))

    def test_old_event_source_is_rejected(self):
        rewards = LiveRewards(); rewards.reset('old')
        with self.assertRaises(ValueError):
            rewards.calculate('old', [dict(id='old', kind='jitter', confirmed=True,
                source='actual_displacement_12f_v1')])

    def test_time_bound_expires_even_with_pauses(self):
        d = MotionJitter()
        for x in [-4, 4]:
            self.assertFalse(d.add(x, 0))
        for _ in range(18):
            self.assertFalse(d.add(0, 0))
        self.assertFalse(d.add(-4, 0))
        self.assertFalse(d.add(4, 0))

import unittest
import numpy as np
from touhou_ai.live_rewards import LiveRewards, WEIGHTS, observed_events, validate_power_gain_upgrade
from touhou_ai.focused_policy import reward_input


def pickup(before=0, after=1, kind=1, identifier='p', bomb=0):
    return dict(id=identifier, kind='power_gain', confirmed=True,
                source='verified_power_pickup_v1', item_type=kind,
                before_raw=before, after_raw=after, amount_raw=after-before, bomb_state=bomb)


class PowerGainRewardTests(unittest.TestCase):
    def test_bomb_pickup_consumed_without_delayed_reward(self):
        r=LiveRewards();r.reset('test')
        event=pickup(0,20,4,bomb=1)
        self.assertEqual(r.calculate('test',[event])[0],0)
        self.assertEqual(r.calculate('test',[event | {'bomb_state':0}])[0],0)
        self.assertAlmostEqual(r.calculate('test',[pickup(20,40,4,'next')])[0],.1)
        for state in (None, True, 2):
            with self.assertRaises(ValueError):
                r.calculate('test',[pickup(identifier='bad',bomb=state)])
    def test_amounts_cap_and_dedup(self):
        for event, expected in [(pickup(), .005), (pickup(0,20,4), .1),
                                (pickup(95,100,11), .025), (pickup(0,1,10), .005)]:
            r = LiveRewards(); r.reset('test')
            total, parts = r.calculate('test', [event])
            self.assertAlmostEqual(total, expected)
            self.assertEqual(parts['power_gain'], total)
            self.assertEqual(r.calculate('test', [event])[0], 0)

    def test_reject_invalid_events(self):
        for change in [dict(source='snapshot_delta'), dict(amount_raw=20),
                       dict(before_raw=-1), dict(after_raw=101), dict(item_type=2),
                       dict(amount_raw=1.), dict(before_raw=1,after_raw=1,amount_raw=0)]:
            r=LiveRewards(); r.reset('test')
            with self.assertRaises(ValueError): r.calculate('test',[pickup() | change])

    def test_only_pickups_not_snapshot_or_respawn_increase(self):
        before=dict(stage=1,stage_frame=10,lives_raw=2,power_raw=0,replay_mode=0,mode_flags=0)
        after=before | dict(stage_frame=12,power_raw=60)
        self.assertEqual(observed_events(before,after),[])
        # Pickup remains counted even if the 2F net Power delta is negative.
        events=observed_events(before | dict(power_raw=60),after | dict(power_raw=0,
            combat_reward_events=[pickup(60,61)]))
        r=LiveRewards();r.reset('test')
        self.assertAlmostEqual(r.calculate('test',events)[0],.005)

    def test_input_shape_and_strict_upgrade(self):
        np.testing.assert_array_equal(reward_input({'power_gain':.1}),reward_input())
        source=dict(reward_version='th10-rewards-v19',
                    reward_weights={k:v for k,v in WEIGHTS.items() if k!='power_gain'})
        validate_power_gain_upgrade(source)
        for change in [dict(reward_version='th10-rewards-v18'),dict(evasion_only=True),
                       dict(reward_weights=WEIGHTS)]:
            with self.assertRaises(ValueError):validate_power_gain_upgrade(source | change)

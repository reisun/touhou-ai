import unittest
from unittest.mock import patch
from touhou_ai.live_rewards import LiveRewards, observed_events, MILESTONES


def damage(identifier='d', power=0, bomb=0, amount=1000):
    return dict(id=identifier, kind='damage', confirmed=True, amount=amount,
                source='verified_game_event', power_raw=power, bomb_state=bomb)


class LiveRewardTests(unittest.TestCase):
    def rewards(self):
        r = LiveRewards(); r.reset('test'); return r

    def test_power_multiplier_and_bomb_boundaries(self):
        for raw, expected in [(0, 5), (20, 7.5), (60, 12.5), (100, 17.5)]:
            r = self.rewards()
            self.assertEqual(r.calculate('test', [damage(power=raw)])[0], expected)
            self.assertEqual(r.calculate('test', [damage(power=raw)])[0], 0)
            self.assertEqual(r.calculate('test', [damage('bomb', raw, 1)])[0], 0)
        before = dict(stage=1, stage_frame=10, lives_raw=2, power_raw=0, replay_mode=0, mode_flags=0)
        after = before | dict(stage_frame=12, bomb={'state': 0}, power_raw=100,
                              combat_reward_events=[damage('first-frame', 20, 1), damage('last-frame', 0, 0)])
        self.assertEqual(self.rewards().calculate('test', observed_events(before, after))[0], 5)

    def test_removed_rewards_and_stage_disappearance_do_not_award(self):
        r = self.rewards()
        kinds = ('kill', 'stage_clear', 'power', 'low_power', 'invalid_bomb')
        for k in kinds:
            with self.assertRaises(ValueError):
                r.calculate('test', [dict(id=k, kind=k, confirmed=True, amount=1)])
        before = dict(stage=1, stage_frame=10, lives_raw=2, power_raw=100, replay_mode=0, mode_flags=0)
        for after in (before | dict(stage=2, stage_frame=1, power_raw=0),
                      before | dict(stage_frame=12, power_raw=0, enemies=[])):
            events = observed_events(before, after, [0, 0, 0, 1])
            self.assertEqual([e['kind'] for e in events], ['power_down'])
            self.assertEqual(self.rewards().calculate('test', events)[1]['power_down'], 0)
        self.assertEqual(r.calculate('test', observed_events(before, before | dict(stage_frame=12, lives_raw=1)))[0], -10)

    def test_unknown_event_fields_fail_closed_transactionally(self):
        for update in ({'bomb_state': None}, {'bomb_state': 2}, {'power_raw': 101},
                       {'power_raw': None}, {'amount': float('nan')}, {'source': 'hp_difference'}):
            r = self.rewards()
            with self.assertRaises(ValueError):
                r.calculate('test', [damage('good'), damage('bad') | update])
            self.assertEqual(r.calculate('test', [damage('good')])[0], 5)

    def test_progress_unverified_production_is_rejected(self):
        e = dict(id='p', kind='progress', confirmed=True, source='guess',
                 stage=1, milestone='boss_defeat', lives_raw=2, power_raw=100)
        with self.assertRaisesRegex(ValueError, 'acceptance'):
            self.rewards().calculate('test', [e])

    def test_verified_ecl_event_uses_captured_state(self):
        e = dict(id='stage2:2553', kind='progress', confirmed=True, source='verified_ecl_progress_v1',
                 stage=2, milestone='midboss_arrival', lives_raw=1, power_raw=21)
        before = dict(stage=2, stage_frame=2552, lives_raw=1, power_raw=21, replay_mode=0, mode_flags=0)
        after = before | dict(stage_frame=2554, power_raw=0, progress_reward_events=[e])
        r = self.rewards()
        _, parts = r.calculate('test', observed_events(before, after))
        self.assertEqual(parts['progress'], 55.25)
        self.assertEqual(r.calculate('test', [e | {'id': 'duplicate'}])[0], 0)

    def test_progress_formula_and_per_play_stage_milestone_dedup(self):
        with patch('touhou_ai.live_rewards.VERIFIED_PROGRESS_SOURCES', {'fixture'}):
            r = self.rewards()
            for stage in (1, 2):
                for name in MILESTONES:
                    e = dict(id=f'{stage}:{name}', kind='progress', confirmed=True,
                             source='fixture', stage=stage, milestone=name, lives_raw=2, power_raw=100)
                    self.assertEqual(r.calculate('test', [e])[0], 105)
                    self.assertEqual(r.calculate('test', [e | {'id': e['id']+'new'}])[0], 0)
            r.reset('test')
            self.assertEqual(r.calculate('test', [e | {'lives_raw': 1, 'power_raw': 60}])[0], 65)

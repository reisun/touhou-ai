import unittest
from unittest.mock import patch
from touhou_ai.live_rewards import LiveRewards, observed_events, MILESTONES


def damage(identifier='d', power=0, bomb=0, amount=1000):
    return dict(id=identifier, kind='damage', confirmed=True, amount=amount,
                source='verified_game_event', power_raw=power, bomb_state=bomb)


class LiveRewardTests(unittest.TestCase):
    def test_uniform_scaling_and_independent_evasion_contract(self):
        from touhou_ai.evasion_only import DeathOnlyRewards
        from touhou_ai.live_rewards import WEIGHTS
        old = dict(damage=15., damage_power=.5, progress=20.,
                   progress_life=30., progress_power=2., hit=-60., power_down=0.)
        for power in (0, 20, 60, 100):
            events = [damage(power=power),
                      dict(id='hit', kind='hit', confirmed=True),
                      dict(id='p', kind='progress', confirmed=True,
                           source='verified_ecl_progress_v1', stage=1,
                           milestone='boss_defeat', lives_raw=2,
                           power_raw=power, bomb_state=0)]
            previous = LiveRewards(weights=old); previous.reset('test')
            expected, parts = previous.calculate('test', events)
            v17 = LiveRewards(weights=WEIGHTS | {'damage': .25, 'progress_power': 2./60}); v17.reset('test')
            actual, scaled = v17.calculate('test', events)
            self.assertAlmostEqual(actual, expected / 60)
            for key in parts:
                self.assertAlmostEqual(scaled[key], parts[key] / 60)
            evasion = DeathOnlyRewards(); evasion.reset('test')
            self.assertEqual(evasion.calculate('test', events)[0], -60)
        self.assertEqual(WEIGHTS['hit'], -1)
        self.assertEqual(WEIGHTS['progress_power'], .1)

    def test_power_upgrade_source_contract(self):
        from touhou_ai.live_rewards import WEIGHTS, validate_power_upgrade
        source = dict(reward_version='th10-rewards-v17', reward_weights={k:v for k,v in WEIGHTS.items() if k not in ('jitter', 'power_gain')} | {'progress_power': 2./60})
        validate_power_upgrade(source)
        for change in (dict(reward_version='th10-rewards-v18'), dict(evasion_only=True),
                       dict(reward_weights=WEIGHTS), dict(reward_weights=source['reward_weights'] | {'hit': -60})):
            with self.assertRaises(ValueError):
                validate_power_upgrade(source | change)

    def rewards(self):
        r = LiveRewards(); r.reset('test'); return r

    def test_power_multiplier_and_bomb_boundaries(self):
        for raw, expected in [(0, 12), (20, 18), (60, 30), (100, 42)]:
            r = self.rewards()
            self.assertAlmostEqual(r.calculate('test', [damage(power=raw)])[0], expected / 60)
            self.assertEqual(r.calculate('test', [damage(power=raw)])[0], 0)
            self.assertEqual(r.calculate('test', [damage('bomb', raw, 1)])[0], 0)
        before = dict(stage=1, stage_frame=10, lives_raw=2, power_raw=0, replay_mode=0, mode_flags=0)
        after = before | dict(stage_frame=12, bomb={'state': 0}, power_raw=100,
                              combat_reward_events=[damage('first-frame', 20, 1), damage('last-frame', 0, 0)])
        self.assertEqual(self.rewards().calculate('test', observed_events(before, after))[0], .2)

    def test_removed_rewards_and_stage_disappearance_do_not_award(self):
        r = self.rewards()
        kinds = ('kill', 'stage_clear', 'power', 'low_power', 'invalid_bomb', 'power_down')
        for k in kinds:
            with self.assertRaises(ValueError):
                r.calculate('test', [dict(id=k, kind=k, confirmed=True, amount=1)])
        before = dict(stage=1, stage_frame=10, lives_raw=2, power_raw=100, replay_mode=0, mode_flags=0)
        for after in (before | dict(stage=2, stage_frame=1, power_raw=0),
                      before | dict(stage_frame=12, power_raw=0, enemies=[])):
            events = observed_events(before, after, [0, 0, 0, 1])
            self.assertEqual(events, [])
            total, components = self.rewards().calculate('test', events)
            self.assertEqual(total, 0)
            self.assertNotIn('power_down', components)
        self.assertEqual(r.calculate('test', observed_events(before, before | dict(stage_frame=12, lives_raw=1)))[0], -1)

    def test_unknown_event_fields_fail_closed_transactionally(self):
        for update in ({'bomb_state': None}, {'bomb_state': 2}, {'power_raw': 101},
                       {'power_raw': None}, {'amount': float('nan')}, {'source': 'hp_difference'}):
            r = self.rewards()
            with self.assertRaises(ValueError):
                r.calculate('test', [damage('good'), damage('bad') | update])
            self.assertEqual(r.calculate('test', [damage('good')])[0], .2)

    def test_progress_unverified_production_is_rejected(self):
        e = dict(id='p', kind='progress', confirmed=True, source='guess',
                 stage=1, milestone='boss_defeat', lives_raw=2, power_raw=100)
        with self.assertRaisesRegex(ValueError, 'acceptance'):
            self.rewards().calculate('test', [e])

    def test_verified_ecl_event_uses_captured_state(self):
        e = dict(id='stage2:2553', kind='progress', confirmed=True, source='verified_ecl_progress_v1',
                 stage=2, milestone='midboss_arrival', lives_raw=1, power_raw=21, bomb_state=0)
        before = dict(stage=2, stage_frame=2552, lives_raw=1, power_raw=21, replay_mode=0, mode_flags=0)
        after = before | dict(stage_frame=2554, power_raw=0, progress_reward_events=[e])
        r = self.rewards()
        _, parts = r.calculate('test', observed_events(before, after))
        self.assertAlmostEqual(parts['progress'], 50 / 60 + .1 * 1.05)
        self.assertEqual(r.calculate('test', [e | {'id': 'duplicate'}])[0], 0)

    def test_progress_formula_and_per_play_stage_milestone_dedup(self):
        with patch('touhou_ai.live_rewards.VERIFIED_PROGRESS_SOURCES', {'fixture'}):
            r = self.rewards()
            for stage in (1, 2):
                for name in MILESTONES:
                    e = dict(id=f'{stage}:{name}', kind='progress', confirmed=True,
                             source='fixture', stage=stage, milestone=name, lives_raw=2, power_raw=100, bomb_state=0)
                    self.assertAlmostEqual(r.calculate('test', [e])[0], 80 / 60 + .1 * 5)
                    self.assertEqual(r.calculate('test', [e | {'id': e['id']+'new'}])[0], 0)
            r.reset('test')
            self.assertAlmostEqual(r.calculate('test', [e | {'lives_raw': 1, 'power_raw': 60}])[0], 50 / 60 + .1 * 3)

    def test_progress_bomb_boundary_and_no_delayed_bonus(self):
        for milestone in MILESTONES:
            e = dict(id='p', kind='progress', confirmed=True,
                     source='verified_ecl_progress_v1', stage=1, milestone=milestone,
                     lives_raw=2, power_raw=100, bomb_state=1)
            before = dict(stage=1, stage_frame=10, lives_raw=2, power_raw=100,
                          replay_mode=0, mode_flags=0)
            after = before | dict(stage_frame=12, bomb={'state': 0}, progress_reward_events=[e])
            r = self.rewards()
            self.assertEqual(r.calculate('test', observed_events(before, after))[0], 0)
            self.assertEqual(r.calculate('test', [e | dict(id='later', bomb_state=0)])[0], 0)
            r.reset('test')
            after.update(bomb={'state': 1}, progress_reward_events=[e | dict(bomb_state=0)])
            self.assertAlmostEqual(r.calculate('test', observed_events(before, after))[0], 80 / 60 + .1 * 5)

    def test_progress_unknown_bomb_state_is_transactional(self):
        e = dict(id='p', kind='progress', confirmed=True,
                 source='verified_ecl_progress_v1', stage=1, milestone='boss_defeat',
                 lives_raw=2, power_raw=100)
        for state in (None, -1, 2, True, 0.0):
            r = self.rewards()
            with self.assertRaisesRegex(ValueError, 'bomb state'):
                r.calculate('test', [damage(), e | dict(bomb_state=state)])
            self.assertEqual(r.calculate('test', [damage()])[0], .2)
            self.assertAlmostEqual(r.calculate('test', [e | dict(bomb_state=0)])[0], 80 / 60 + .1 * 5)

class ShotUpdateTests(unittest.TestCase):
    def test_only_exact_v25_migrates(self):
        from touhou_ai.live_rewards import WEIGHTS, validate_shot_update
        old=dict(reward_version='th10-rewards-v25',reward_weights=dict(WEIGHTS,damage=.25))
        validate_shot_update(old)
        for change in [dict(reward_version='th10-rewards-v24'),dict(evasion_only=True),dict(reward_weights=WEIGHTS)]:
            with self.assertRaises(ValueError):validate_shot_update(old|change)

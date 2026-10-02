import unittest
import numpy as np
from touhou_ai.vibration_reward import VibrationReward, vibration, vibration_components, SOURCE
from touhou_ai.live_rewards import LiveRewards
from touhou_ai.live_rewards import validate_vibration_upgrade, WEIGHTS

class VibrationTests(unittest.TestCase):
    def test_regular_taps_have_constant_average_velocity(self):
        tap = vibration_components([(4,0),(0,0)]*9)
        shake = vibration_components([(4,0),(-4,0)]*9)
        self.assertAlmostEqual(tap['averaged_velocity_residual'],0)
        self.assertGreater(tap['vibration'],0)
        self.assertAlmostEqual(tap['vibration']*2,shake['vibration'])
        self.assertLess(tap['vibration'],1.1)
        slower = vibration_components(([(4,0)]*2+[(0,0)]*2)*4+[(4,0)]*2)
        self.assertAlmostEqual(slower['averaged_velocity_residual'],0)
        self.assertLess(slower['vibration'],1.5)

    def test_rotation_drift_and_phase_invariance(self):
        base=np.array([(4,0),(0,0)]*9,dtype=float)
        r=np.array([[.6,-.8],[.8,.6]])
        score=vibration(base)
        self.assertAlmostEqual(score,vibration(base@r))
        self.assertAlmostEqual(score,vibration(base+[2,3]))
        self.assertAlmostEqual(score,vibration(base[::-1]))
        self.assertAlmostEqual(score*2,vibration(base*2))

    def test_independent_position_fit(self):
        rng=np.random.default_rng(20)
        for _ in range(40):
            v=rng.normal(size=(18,2))*4
            pos=np.vstack([np.zeros((1,2)),np.cumsum(v,axis=0)])
            design=np.column_stack([np.ones(19),np.arange(19)])
            fitted=design@np.linalg.lstsq(design,pos,rcond=None)[0]
            expected=np.sqrt(np.mean(np.sum((pos-fitted)**2,axis=1)))
            self.assertAlmostEqual(vibration_components(v)['position_rms_pixels'],expected)

    def test_migration_rejects_other_contracts(self):
        manifest=dict(reward_version='th10-rewards-v23',reward_weights=WEIGHTS,
                      jitter_detector=dict(version='actual_displacement_vibration_36f_v1'))
        validate_vibration_upgrade(manifest)
        for change in [dict(reward_version='th10-rewards-v22'),dict(reward_weights={}),
                       dict(evasion_only=True),dict(jitter_detector={})]:
            with self.assertRaises(ValueError):validate_vibration_upgrade(manifest|change)

    def test_straight_single_turn_and_single_stop(self):
        for path in ([(4,0)]*60, [(0,0)]*60,
                     [(4,0)]*25+[(-4,0)]*25,
                     [(4,0)]*20+[(0,0)]*20+[(4,0)]*20):
            d=VibrationReward()
            self.assertEqual(sum(d.add(*v) for v in path),0)

    def test_shake_is_proportional_and_bounded(self):
        sums=[]
        for scale in (.5,1.):
            d=VibrationReward()
            sums.append(sum(d.add(4*scale,4*scale if i%2 else -4*scale) for i in range(120)))
        self.assertGreater(sums[0],0)
        self.assertAlmostEqual(sums[1],sums[0]*2)
        d=VibrationReward();values=[d.add(9 if i%2 else -9,0) for i in range(200)]
        for i in range(len(values)-30):self.assertLessEqual(sum(values[i:i+30]),1.00000001)

    def test_no_tail_charge_on_stop_or_steady_motion(self):
        for tail in [(0,0),(4,4)]:
            d=VibrationReward()
            for i in range(40):d.add(4,4 if i%2 else -4)
            # First transition to steady movement may be a movement change.
            d.add(*tail)
            self.assertEqual(sum(d.add(*tail) for _ in range(30)),0)
        d=VibrationReward()
        for i in range(40):d.add(4,4 if i%2 else -4)
        self.assertEqual(d.add(0,0),0)

    def test_gap_resets_history(self):
        d=VibrationReward()
        for i in range(40):d.add(4,4 if i%2 else -4)
        a=dict(player=dict(status=1,position=[0,100]),stage=1,stage_frame=2,lives_raw=2)
        self.assertFalse(d.observe(a,a|dict(stage_frame=6)))
        self.assertEqual(len(d.moves),0)

    def test_fraction_validation_and_reward(self):
        r=LiveRewards();r.reset('t')
        e=dict(id='j',kind='jitter',confirmed=True,source=SOURCE,amount=1./30)
        reward,parts=r.calculate('t',[e]);self.assertEqual(reward,0);self.assertNotIn('jitter',parts)
        self.assertEqual(r.calculate('t',[e])[0],0)
        for amount in [None,float('nan'),-.1,.1,True]:
            r.reset('t')
            with self.assertRaises(ValueError):r.calculate('t',[e|dict(amount=amount)])

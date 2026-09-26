import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from touhou_ai.obs_stats import ObsStats
from touhou_ai.live_rewards import VERSION
from touhou_ai.bullet_scope import SPEC


class ObsStatsTests(unittest.TestCase):
    def test_new_grid_campaign_does_not_mix_focused_growth(self):
        from touhou_ai.dual_grid import SPEC as GRID_SPEC, CONTRACT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, scope, contract, score in [('live-learning-01', SPEC, 'focused', -10),
                                                  ('live-learning-02', GRID_SPEC, CONTRACT, 5)]:
                run = root/name; run.mkdir()
                (run/'status.json').write_text(json.dumps({'backend': 'real_th10',
                    'contract': contract, 'reward_version': VERSION, 'bullet_scope': scope,
                    'episodes': [{'reload_verified': True, 'return': score}]}))
            stats = ObsStats(root).snapshot()
            self.assertEqual([e['return'] for e in stats['growth']], [5])

    def test_old_contract_history_is_not_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'live-learning-old'
            run.mkdir()
            (run / 'status.json').write_text(json.dumps({'backend': 'real_th10',
                'reward_version': 'th10-rewards-v2', 'bullet_scope': SPEC,
                'episodes': [{'reload_verified': True, 'return': -10}]}))
            stats = ObsStats(root).snapshot()
            self.assertEqual(stats['growth'], [])
            self.assertIn('damage', stats['totals'])
            self.assertIn('progress', stats['totals'])

    def test_synchronized_window_partial_line_and_no_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'live-learning-test'
            run.mkdir()
            (run / 'status.json').write_text(json.dumps({'backend': 'real_th10', 'episodes': [],
                                                       'reward_version': VERSION, 'bullet_scope': SPEC}))
            def row(t, value, reward, dead=False):
                return json.dumps({'raw': {'lives_raw': -1 if dead else 2}, 'telemetry': {
                    'timestamp': t, 'episode_id': 'one', 'policy': {'value': value},
                    'model': {'ppo': {'gamma': .99}, 'reward_weights': {'hit': -5}},
                    'reward': {'total': reward, 'components': {'hit': reward}, 'enabled': ['hit']}}})
            path = run / 'episode-1.jsonl'
            path.write_text(row(69, 0, -5)+'\n'+row(80, 2, -5)+'\n'+row(90, 1, 0, True)+'\n'+row(99, 1, 0))
            store = ObsStats(root)
            with patch('touhou_ai.obs_stats.time.time', return_value=100):
                a = store.snapshot()
                b = store.snapshot()
            self.assertEqual(a, b)
            self.assertEqual(a['samples'], 2)
            self.assertEqual(a['totals'], {'hit': -5})
            self.assertAlmostEqual(a['points'][0]['td'], -6.01)
            self.assertEqual(a['points'][1]['td'], -1)
            self.assertEqual(a['end']-a['start'], 30)
            with path.open('a') as stream:
                stream.write('\n')
            with patch('touhou_ai.obs_stats.time.time', return_value=100):
                self.assertEqual(store.snapshot()['samples'], 3)
            with patch('touhou_ai.obs_stats.time.time', return_value=131):
                self.assertEqual(store.snapshot()['samples'], 0)

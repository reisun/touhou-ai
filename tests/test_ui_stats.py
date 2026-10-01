import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from touhou_ai.ui_stats import UiStats
from touhou_ai.obs_stats import ObsStats
from touhou_ai.live_rewards import VERSION
from touhou_ai.dual_grid import SPEC


class UiStatsTests(unittest.TestCase):
    def test_memory_window_td_expiry_growth_and_no_raw_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'live-learning-test';run.mkdir()
            (run/'status.json').write_text(json.dumps({'backend':'real_th10',
                'reward_version':VERSION,'bullet_scope':SPEC,'ui_stats_transport':'shared_memory_v1',
                'episodes':[{'episode':1,'reload_verified':True,'return':-4,'max_progress':{'rank':3}}]}))
            stats=UiStats(run)
            def add(t,episode,value,reward,dead=False):
                stats.add({'timestamp':t,'policy':{'value':value},'model':{'reward_weights':{'hit':-5}},
                    'reward':{'total':reward,'components':{'hit':reward},'enabled':['hit']}},
                    [],dead,episode,.99)
            with patch('touhou_ai.ui_stats.publish',return_value=True) as publish:
                add(69,'one',0,1)
                add(80,'one',2,-5)
                add(90,'one',1,0,True)
                stats.begin_episode()
                add(99,'two',3,1)
                stats.flush()
                payload=publish.call_args.args[0]
            self.assertNotIn('raw',json.dumps(payload))
            store=ObsStats(root)
            with patch('touhou_ai.obs_stats.read_memory',return_value=(1,json.dumps(payload).encode())), \
                 patch.object(store,'episode_progress',side_effect=AssertionError('full log scan')), \
                 patch('touhou_ai.obs_stats.time.time',return_value=100):
                result=store.snapshot()
                self.assertEqual(result,store.snapshot())
                self.assertEqual(result['samples'],3)
                self.assertEqual(result['totals'],{'hit':-4})
                self.assertAlmostEqual(result['points'][0]['td'],-6.01)
                self.assertEqual(result['points'][1]['td'],-1)
                self.assertIsNone(result['points'][2]['td'])
                self.assertEqual(result['growth'][0]['max_progress'],{'rank':3})
            with patch('touhou_ai.obs_stats.read_memory',return_value=None), \
                 patch('touhou_ai.obs_stats.time.time',return_value=131):
                self.assertEqual(store.snapshot()['samples'],0)
            self.assertEqual(list(run.glob('episode-*.jsonl')),[])

    def test_progress_validated_max_and_reset(self):
        from touhou_ai.progress_schema import PROGRESS_AXIS
        with tempfile.TemporaryDirectory() as directory, patch('touhou_ai.ui_stats.publish',return_value=True):
            stats=UiStats(Path(directory)/'run')
            good={'kind':'progress','confirmed':True,'source':'verified_ecl_progress_v2',
                  'difficulty_raw':1,'stage':1,'milestone':'boss_arrival'}
            t={'timestamp':100,'policy':{'value':0},'reward':{'total':1,'components':{'progress':1},'enabled':['progress']}}
            stats.add(t,[good],False,'one',.99)
            self.assertEqual(stats.best['rank'],3)
            stats.add(t,[good|{'milestone':'boss_defeat','confirmed':False}],False,'one',.99)
            self.assertEqual(stats.best['rank'],3)
            stats.begin_episode();self.assertIsNone(stats.best)

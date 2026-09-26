import json
from pathlib import Path
import tempfile
import unittest
from touhou_ai.obs_stats import ObsStats
from touhou_ai.live_rewards import VERSION
from touhou_ai.bullet_scope import SPEC


class GrowthProgressTests(unittest.TestCase):
    def test_run_episode_join_missing_and_non_cumulative_maximum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('live-learning-a', 'live-learning-b'):
                run = root/name
                run.mkdir()
                (run/'status.json').write_text(json.dumps(dict(backend='real_th10',
                    reward_version=VERSION, bullet_scope=SPEC, episodes=[
                        dict(episode=n, reload_verified=True, **{'return': n}) for n in (3, 1, 2)])))
                for episode, stage, milestone in ((1, 2, 'midboss_arrival'), (3, 1, 'boss_defeat')):
                    if name.endswith('b'):
                        stage = 6
                    event = dict(kind='progress', confirmed=True, source='verified_ecl_progress_v1',
                                 stage=stage, milestone=milestone)
                    row = dict(telemetry=dict(episode_id=f'{name}-{episode}'), events=[event])
                    (run/f'episode-{episode}.jsonl').write_text(json.dumps(row)+'\n')
            result = ObsStats(root).snapshot()['growth']
            self.assertEqual([e['max_progress']['rank'] if e['max_progress'] else None
                              for e in result], [4, 5, None, 24, 21, None])

    def test_validation_maximum_partial_append_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'live-learning-test'
            run.mkdir()
            path = run/'episode-1.jsonl'
            def row(stage, milestone='boss_defeat', source='verified_ecl_progress_v1', identity=None):
                return json.dumps(dict(telemetry=dict(episode_id=identity or run.name+'-1'),
                    events=[dict(kind='progress', confirmed=True, source=source,
                                 stage=stage, milestone=milestone)]))+'\n'
            path.write_text(row(1)+row(6, source='fixture')+row(6, identity='wrong-1')+
                            row(True)+row(7)+row(1, 'unknown')+row(2, 'midboss_arrival')+
                            row(1)+row(3).rstrip())
            store = ObsStats(run.parent)
            self.assertEqual(store.episode_progress(run, 1)['rank'], 5)
            self.assertEqual(store.episode_progress(run, 1)['rank'], 5)
            with path.open('a') as stream:
                stream.write('\n')
            self.assertEqual(store.episode_progress(run, 1)['rank'], 12)
            self.assertIsNone(store.episode_progress(run, None))
            self.assertIsNone(store.episode_progress(run, 2))


if __name__ == '__main__':
    unittest.main()

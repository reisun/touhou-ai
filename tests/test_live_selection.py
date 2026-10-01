import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from touhou_ai.obs_stats import ObsStats
class LiveSelectionTests(unittest.TestCase):
 def test_switch_unknown_version_no_scan_and_no_unchanged_reads(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);root=base/'artifacts';root.mkdir();runtime=base/'.runtime';runtime.mkdir();record=runtime/'live-learning.json'
   def make(name,version):
    p=root/name;p.mkdir();(p/'status.json').write_text(json.dumps(dict(backend='real_th10',reward_version=version,reward_weights={'hit':-60},reward_enabled=['hit'],ui_stats_transport='shared_memory_v1',episodes=[dict(episode=1,reload_verified=True,max_progress={'rank':1})])));return p
   old=make('live-learning-z','old');new=make('live-learning-a','unknown-future');record.write_text(json.dumps({'RunId':old.name}));store=ObsStats(root)
   with patch('touhou_ai.obs_stats.read_memory',return_value=None),patch.object(Path,'glob',side_effect=AssertionError('managed LIVE must not scan')):
    self.assertEqual(store.snapshot()['active_run'],old.name)
    store.next_manifest_check=0
    with patch.object(Path,'read_text',side_effect=AssertionError('unchanged files must not be read')):
     for _ in range(100):self.assertEqual(store.snapshot()['active_run'],old.name)
    record.write_text(json.dumps({'RunId':new.name}));store.next_manifest_check=0
    result=store.snapshot();self.assertEqual(result['active_run'],new.name);self.assertEqual(result['reward_version'],'unknown-future');self.assertEqual(len(result['growth']),1)
    record.write_text(json.dumps({'RunId':'live-learning-pending'}));store.next_manifest_check=0
    result=store.snapshot();self.assertIsNone(result['active_run']);self.assertEqual(result['growth'],[])
if __name__=='__main__':unittest.main()

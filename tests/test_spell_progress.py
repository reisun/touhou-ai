import json
from pathlib import Path
import re
import tempfile
import unittest
from touhou_ai.progress_schema import SPELLS, PROGRESS_AXIS, progress_point
from touhou_ai.live_rewards import LiveRewards
from touhou_ai.obs_stats import ObsStats


def event(spell, bomb=0, identifier=None):
    return dict(id=identifier or f"spell:{spell['id']}", kind='progress', confirmed=True,
                source='verified_ecl_progress_v2', stage=spell['stage'], difficulty_raw=1,
                milestone=spell['role']+'_defeat' if spell['final'] else 'spell_breakthrough',
                spell_id_raw=spell['id'], lives_raw=2, power_raw=100, bomb_state=bomb)


class SpellProgressTests(unittest.TestCase):
    def test_all_normal_spells_match_reviewed_ecl(self):
        root = Path(__file__).resolve().parents[1]
        reference = json.loads((root/'docs/spell-reference-normal.json').read_text(encoding='utf-8'))['spells']
        self.assertEqual(len(SPELLS), 24)
        for spell, row in zip(SPELLS, reference):
            self.assertEqual((spell['id'], spell['stage'], spell['role'], spell['final']),
                             (row['spell_id_raw'], row['stage'], row['role'], row['final_for_encounter']))
            source = (root/row['source_file']).read_text(encoding='cp932')
            sub = 'MBoss' if spell['id'] == 11 else row['subroutine']
            body = re.search(r'void '+sub+r'\([^\n]*\)\n\{(.*?)\n\}', source, re.S).group(1)
            pattern = r'ins_334\(\d+, '+str(spell['hp'])+r', \d+, "'+spell['callback']+r'"\)'
            self.assertRegex(body, pattern)

    def test_each_breakthrough_and_final_awarded_once(self):
        rewards = LiveRewards(); rewards.reset('test')
        for spell in SPELLS:
            e = event(spell)
            self.assertAlmostEqual(rewards.calculate('test', [e])[0], 20/60 + .5*2 + .1*5)
            self.assertEqual(rewards.calculate('test', [e | {'id': e['id']+'duplicate'}])[0], 0)
            if spell['final']:
                self.assertEqual(rewards.calculate('test', [e | {'id': 'boss:'+e['id'], 'spell_id_raw': None}])[0], 0)
        self.assertEqual(len([p for p in PROGRESS_AXIS if p['milestone']=='spell_breakthrough']), 15)

    def test_bomb_zero_still_records_progress_and_no_later_award(self):
        e = event(SPELLS[0], bomb=1)
        rewards = LiveRewards(); rewards.reset('test')
        self.assertEqual(rewards.calculate('test', [e])[0], 0)
        self.assertEqual(rewards.calculate('test', [e | {'id': 'later', 'bomb_state': 0}])[0], 0)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'run'; run.mkdir()
            row = dict(telemetry={'episode_id':'run-1'}, events=[e])
            (run/'episode-1.jsonl').write_text(json.dumps(row)+'\n')
            point = ObsStats(run.parent).episode_progress(run, 1)
            self.assertEqual(point['label'], '1面 スペル突破')
            self.assertTrue(3 < point['rank'] < 4)
            row['events'].append(event(SPELLS[1]))
            (run/'episode-1.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(ObsStats(run.parent).episode_progress(run, 1)['rank'], 4)

    def test_reject_wrong_difficulty_stage_final_or_missing_spell(self):
        e = event(SPELLS[0])
        for change in ({'difficulty_raw': 2}, {'difficulty_raw': True}, {'stage': 2},
                       {'spell_id_raw': 7}, {'spell_id_raw': None}, {'spell_id_raw': 999},
                       {'source': 'verified_ecl_progress_v1'}, {'lives_raw': -1}):
            r = LiveRewards(); r.reset('test')
            with self.assertRaises(ValueError): r.calculate('test', [e | change])
            self.assertAlmostEqual(r.calculate('test', [e])[0], 20/60 + .5*2 + .1*5)

    def test_final_spell_cannot_impersonate_another_milestone(self):
        e = event(SPELLS[1])
        for change in ({'milestone': 'boss_arrival'}, {'stage': 2}, {'spell_id_raw': 3}, {'difficulty_raw': 2}):
            with self.assertRaises(ValueError): progress_point(e | change)

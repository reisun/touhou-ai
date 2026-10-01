"""Normal-only pinned ECL progression shared with native hooks and OBS."""
import json
from pathlib import Path

SPELLS = json.loads(Path(__file__).with_name('spell_progress.json').read_text())
SPELL_BY_ID = {s['id']: s for s in SPELLS}
MILESTONES = ('midboss_arrival', 'midboss_defeat', 'boss_arrival', 'boss_defeat')
PROGRESS_AXIS = []
for stage in range(1, 7):
    for index, (milestone, label) in enumerate(zip(MILESTONES,
            ('中ボス着', '中ボス突破', 'ボス着', 'ボス突破')), 1):
        rank = (stage-1)*4 + index
        PROGRESS_AXIS.append(dict(stage=stage, milestone=milestone, rank=rank,
                                  label=f'{stage}面 {label}'))
        if milestone == 'boss_arrival':
            spells = [s for s in SPELLS if s['stage'] == stage and not s['final']]
            for n, spell in enumerate(spells, 1):
                PROGRESS_AXIS.append(dict(stage=stage, milestone='spell_breakthrough',
                    spell_id_raw=spell['id'], rank=rank+n/(len(spells)+1),
                    label=f'{stage}面 スペル突破'))


def progress_point(event):
    stage, milestone = event.get('stage'), event.get('milestone')
    if type(stage) is not int or not 1 <= stage <= 6:
        raise ValueError('invalid milestone stage')
    if event.get('source') == 'verified_ecl_progress_v2':
        if type(event.get('difficulty_raw')) is not int or event['difficulty_raw'] != 1:
            raise ValueError('invalid progress difficulty')
    if milestone == 'spell_breakthrough':
        identifier = event.get('spell_id_raw')
        spell = SPELL_BY_ID.get(identifier) if type(identifier) is int else None
        if (not spell or spell['stage'] != stage or spell['final']
                or type(event.get('difficulty_raw')) is not int or event['difficulty_raw'] != 1
                or event.get('source') != 'verified_ecl_progress_v2'):
            raise ValueError('invalid spell breakthrough')
        return next(p for p in PROGRESS_AXIS if p.get('spell_id_raw') == identifier)
    if milestone not in MILESTONES:
        raise ValueError('invalid milestone')
    if event.get('source') == 'verified_ecl_progress_v2' and event.get('spell_id_raw') is not None:
        identifier = event['spell_id_raw']
        spell = SPELL_BY_ID.get(identifier) if type(identifier) is int else None
        if (not spell or not spell['final'] or spell['stage'] != stage
                or milestone != spell['role']+'_defeat'):
            raise ValueError('invalid final spell milestone')
    return next(p for p in PROGRESS_AXIS if p['stage'] == stage and p['milestone'] == milestone)

"""Replay real diagnostic logs; never enables a production reward source."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from touhou_ai.live_rewards import LiveRewards, MILESTONES, VERIFIED_PROGRESS_SOURCES
from touhou_ai.progress_evidence import classify


def main():
    root = Path('artifacts')
    results = {}
    for suffix in ('b', 'd', 'e'):
        directory = root / f'progress-probe-20260927-{suffix}'
        with (directory / 'trace.jsonl').open(encoding='utf-8') as stream:
            result = classify((json.loads(line) for line in stream),
                              json.loads((directory / 'ecl.json').read_text(encoding='utf-8')))
        (directory / 'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results[suffix] = result

    assert {e['milestone'] for r in results.values() for e in r['milestones']} == set(MILESTONES)
    assert [(e['milestone'], e['frame']) for e in results['e']['milestones']] == [
        ('midboss_arrival', 2521), ('boss_arrival', 4925), ('boss_defeat', 8590)]
    timeout = next(e for e in results['e']['excluded'] if e['name'] == 'MBossEscape')
    assert timeout['reason'] == 'timeout' and timeout['hp_before'] == 8400 and timeout['hp'] == 0
    phase = next(e for e in results['e']['excluded'] if e['name'] == 'Boss2')
    assert phase['hp'] == 0 and phase['reason'] == 'phase_or_other_callback'

    # Exercise the existing formula/dedup with event-time data in a scoped fixture.
    # This patch exists only in this offline process, not in the learning runtime.
    with patch('touhou_ai.live_rewards.VERIFIED_PROGRESS_SOURCES', {'offline_acceptance_fixture'}):
        for suffix, result in results.items():
            rewards = LiveRewards()
            rewards.reset(suffix)
            for event in result['milestones']:
                accepted = event | {'id': f"{suffix}:{event['frame']}", 'kind': 'progress',
                                    'confirmed': True, 'source': 'offline_acceptance_fixture'}
                total, parts = rewards.calculate(suffix, [accepted])
                assert total == event['prospective_reward'] == parts['progress']
                assert rewards.calculate(suffix, [accepted | {'id': accepted['id'] + '-duplicate'}])[0] == 0
    assert VERIFIED_PROGRESS_SOURCES == {'verified_ecl_progress_v1'}

    production = []
    rewards = LiveRewards()
    rewards.reset('production')
    with (root / 'progress-probe-20260927-f' / 'trace.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            for event in row['state']['progress_reward_events']:
                diagnostic = [e for e in row['progress_probe'] if e['kind'] in ('instruction', 'callback')
                              and e['frame'] == event['frame'] and e['stage'] == event['stage']]
                assert diagnostic
                assert all(event[k] == diagnostic[0][k] for k in ('lives_raw', 'power_raw'))
                total, _ = rewards.calculate('production', [event])
                assert total == 20 + 30 * event['lives_raw'] + event['power_raw'] / 4
                assert rewards.calculate('production', [event | {'id': event['id'] + '-duplicate'}])[0] == 0
                production.append(event | {'reward': total})
    assert {'midboss_arrival', 'midboss_defeat'} <= {e['milestone'] for e in production}

    checkpoint = root / 'live-learning-20260927-004504-49b5dc' / 'real-episode-5.zip'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == '930ec17959a33b14c93d7cb496d05e2c2b6448c0f4aa7e81f1eec3ef9e8aaca1'
    summary = {'status': 'passed', 'scope': 'stage 1, Normal, Reimu B',
               'live_milestones_verified': list(MILESTONES),
               'timeout_excluded': True, 'zero_hp_phase_excluded': True,
               'event_time_formula_and_dedup': True, 'checkpoint_sha256': digest,
               'production_progress_enabled': True, 'training_updates': 0,
               'production_events': production,
               'cleanup_note': 'Run e crossed stage 2 loading; pause failed during loading. '
                               'A subsequent neutral reattachment confirmed stage 2 game-over, input zero. '
                               'Probe cleanup now bridges loading before pause.',
               'events': {k: [{n: e[n] for n in ('milestone', 'frame', 'lives_raw', 'power_raw', 'prospective_reward')}
                              for e in r['milestones']] for k, r in results.items()},
               'ecl_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root / 'progress-ecl-20260927').glob('*.ecl'))}}
    target = root / 'progress-validation-20260927.json'
    target.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

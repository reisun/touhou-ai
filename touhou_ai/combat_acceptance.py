"""Bounded real-game combat event checks; never updates a learning model."""
import argparse
from collections import Counter
import json
from pathlib import Path

from touhou_ai.live_runtime import LiveRuntime
from touhou_ai.live_reset import continue_episode, start_episode
from touhou_ai.live_acceptance import pause


def run(output, pid):
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'events': {}, 'samples': 0, 'training_updates': 0}
    runtime = LiveRuntime(pid)
    try:
        state = runtime.snapshot(full=False)
        if state['lives_raw'] == -1 and state['player'] is not None:
            continue_episode(runtime)
        else:
            start_episode(runtime)
        counts = Counter()
        damage = 0
        hp_checks = 0
        with (output / 'trace.jsonl').open('x', encoding='utf-8') as stream:
            for index in range(900):
                before = runtime.snapshot()
                mode = 'neutral' if index < 300 else 'shoot' if index < 600 else 'bomb'
                mask = 0 if mode == 'neutral' else 3 if index == 600 else 1
                after = runtime.step_gameplay(mask, 2)
                events = after['combat_reward_events']
                stream.write(json.dumps({'mode': mode, 'before': before,
                                         'after': after}) + '\n')
                for event in events:
                    counts[mode + ':' + event['kind']] += 1
                    if event['kind'] == 'damage':
                        damage += event['amount']
                # Independently check surviving non-boss HP against event amounts.
                old = {e['address']: e for e in before['enemies'] or []}
                for enemy in after['enemies'] or []:
                    previous = old.get(enemy['address'])
                    if previous and not enemy['is_boss'] and 0 < enemy['hp'] < previous['hp']:
                        actual = sum(e['amount'] for e in events if e['kind'] == 'damage'
                                     and e['address'] == enemy['address'])
                        if actual != previous['hp'] - enemy['hp']:
                            raise AssertionError('damage events do not match surviving enemy HP')
                        hp_checks += 1
                report['samples'] = index + 1
                if index == 600:
                    report['bomb_activated'] = after['bomb']['state'] == 1
                if index % 120 == 0:
                    print(json.dumps({'mode': mode, 'frame': after['stage_frame'],
                                      'events': dict(counts)}), flush=True)
                if after['lives_raw'] < 0:
                    break
        report.update(events=dict(counts), damage=damage, hp_checks=hp_checks)
        if counts['neutral:damage'] or counts['neutral:kill']:
            raise AssertionError('neutral play generated combat reward')
        if not counts['shoot:damage'] or not counts['shoot:kill'] or not hp_checks:
            raise AssertionError('insufficient real combat evidence')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        try:
            state = runtime.snapshot(full=False)
            if state['lives_raw'] >= 0 and state.get('player') and state['pause_words'][1] == 0:
                report['paused_on_exit'] = pause(runtime)['pause_words'][1] == 2
        finally:
            runtime.close()
            (output / 'status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.pid)

"""Offline classification of diagnostic evidence; not a production reward source."""
import argparse
import json
from pathlib import Path


def classify(trace, inventory):
    arrivals = {}
    for sub in inventory:
        if sub['name'] not in ('MBoss', 'Boss'):
            continue
        for instruction in sub['instructions']:
            if (instruction['opcode'] == 0x14c and instruction['time'] == 0
                    and instruction['flags'] == 0 and instruction['args'] == '00000000'):
                arrivals[instruction['address']] = ('midboss_arrival' if sub['name'] == 'MBoss'
                                                     else 'boss_arrival')
    milestones, excluded, seen, owners = [], [], set(), {}
    for row in trace:
        for event in row['progress_probe']:
            if event['stage'] != 1 or row['state']['difficulty'] != 1:
                continue  # Only the stage-1 Normal ECL has been reviewed.
            milestone = None
            if event['kind'] == 'instruction' and event['opcode'] == 0x14c and event['arg0'] == 0:
                milestone = arrivals.get(event['address'])
                if milestone:
                    owners[event['owner']] = milestone.replace('_arrival', '_defeat')
            elif event['kind'] == 'callback':
                final = {'MBossDead': 'midboss_defeat', 'BossDead': 'boss_defeat'}.get(event['name'])
                if event['flags'] & 0x10000:
                    excluded.append(event | {'reason': 'timeout'})
                elif (final and owners.get(event['owner']) == final
                      and event['hp'] == 0 and event['hp_before'] <= 0):
                    milestone = final
                else:
                    excluded.append(event | {'reason': 'phase_or_other_callback'})
            if milestone:
                key = (event['stage'], milestone)
                if key in seen:
                    continue
                seen.add(key)
                milestones.append(event | {'milestone': milestone,
                    'prospective_reward': 20 + 30 * event['lives_raw'] + 5 * event['power_raw'] / 20})
    return {'milestones': milestones, 'excluded': excluded, 'production_rewards_enabled': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    with (args.directory / 'trace.jsonl').open(encoding='utf-8') as stream:
        result = classify((json.loads(line) for line in stream),
                          json.loads((args.directory / 'ecl.json').read_text(encoding='utf-8')))
    (args.directory / 'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

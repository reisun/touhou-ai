"""Read-only campaign analysis; writes only the requested analysis report."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def analyze(path):
    status = json.loads((path / 'status.json').read_text(encoding='utf-8'))
    results = []
    for episode in status['episodes']:
        number = episode['episode']
        initial = json.loads((path / f'initial-state-{number}.json').read_text(encoding='utf-8'))
        prev = initial
        totals = collections.Counter()
        groups = collections.defaultdict(list)
        counts = collections.Counter()
        power, positions, shot, bomb = [], [], [], []
        first_hit = None
        elapsed = 0.
        high_bins = collections.defaultdict(lambda: [0, 0, 0.])
        with (path / f'episode-{number}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                raw, t = row['raw'], row['telemetry']
                policy = t.get('policy')
                totals.update(t.get('reward', {}).get('components', {}))
                elapsed += 2 / 60
                events = row.get('events', [])
                if first_hit is None and any(e['kind'] == 'hit' for e in events):
                    first_hit = elapsed
                counts['kills'] += sum(e['kind'] == 'kill' for e in events)
                counts['damage'] += sum(e.get('amount', 0) for e in events if e['kind'] == 'damage')
                if policy and prev and prev.get('player'):
                    p = prev['power_raw'] / 20
                    active = prev.get('bomb', {}).get('state', 0) != 0
                    probability = policy['bomb']
                    group = 'active' if active else ('available' if p >= 1 else 'unavailable')
                    groups[group].append(probability)
                    power.append(p)
                    x, y = prev['player']['position']
                    positions.append((x, y))
                    shot.append(policy['action'][1])
                    bomb.append(policy['action'][3])
                    counts['p0'] += p == 0
                    counts['p4'] += p >= 4
                    counts['edge'] += abs(x) > 160
                    counts['shot_off'] += not policy['action'][1]
                    counts['bomb_activation'] += not active and raw.get('bomb', {}).get('state', 0) != 0
                    counts['invalid_selected'] += group == 'unavailable' and policy['action'][3] == 1
                    counts['high_unavailable'] += group == 'unavailable' and probability >= .9
                    if group == 'unavailable':
                        cell = high_bins[f'{int((x+192)//48)},{int(y//56)}']
                        cell[0] += 1
                        cell[1] += probability >= .9
                        cell[2] += probability
                prev = raw
        results.append(dict(episode=number, updates_before=episode['total_updates']-1,
            seconds=elapsed, first_hit=first_hit, hits=episode['hits'],
            return_=episode['return'], terminated=episode['terminated'],
            reward=dict(totals), counts=dict(counts), mean_power=statistics.mean(power),
            shot_fraction=statistics.mean(shot), bomb_fraction=statistics.mean(bomb),
            groups={k:dict(n=len(v), mean=statistics.mean(v), maximum=max(v),
                           high=sum(x>=.9 for x in v)) for k,v in groups.items()},
            high_bins=dict(high_bins)))
        if number % 20 == 0:
            print(f'Analyzed {number} episodes', flush=True)
    return dict(campaign=path.name, updates=status['updates'],
                stage_transitions=status.get('stage_transitions', []),
                recoveries=status.get('recoveries', []), episodes=results)


def summarize(rows):
    count = len(rows)
    keys = rows[0]['reward'].keys()
    counts = sum((collections.Counter(r['counts']) for r in rows), collections.Counter())
    n = sum(sum(g['n'] for g in r['groups'].values()) for r in rows)
    groups = {}
    for k in ('unavailable', 'available', 'active'):
        vals = [r['groups'][k] for r in rows if k in r['groups']]
        total = sum(v['n'] for v in vals)
        groups[k] = dict(n=total, mean=sum(v['n']*v['mean'] for v in vals)/max(total,1),
                         high=sum(v['high'] for v in vals))
    return dict(n=count, seconds=statistics.mean(r['seconds'] for r in rows),
        first_hit=statistics.mean(r['first_hit'] for r in rows if r['first_hit'] is not None),
        return_=statistics.mean(r['return_'] for r in rows),
        mean_power=statistics.mean(r['mean_power'] for r in rows),
        shot_fraction=statistics.mean(r['shot_fraction'] for r in rows),
        reward={k:statistics.mean(r['reward'][k] for r in rows) for k in keys},
        per_episode={k:v/count for k,v in counts.items() if k in ('damage','kills','bomb_activation')},
        p0_fraction=counts['p0']/n, p4_fraction=counts['p4']/n, edge_fraction=counts['edge']/n,
        groups=groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.campaign)
    rows = result['episodes']
    result['windows'] = {f'{a}-{b}':summarize([r for r in rows if a<=r['episode']<=b])
                         for a,b in [(1,20),(21,40),(41,60),(61,80),(81,100),(101,122)]}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['windows'], indent=2))

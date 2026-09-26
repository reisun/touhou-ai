"""Stream completed campaign logs; no game access and no model changes."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics as stats
import time


def mean(values):
    values = [v for v in values if v is not None]
    return stats.mean(values) if values else None


def analyze(campaign, output):
    output.mkdir(parents=True, exist_ok=False)
    status = json.loads((campaign/'status.json').read_text())
    result = {'campaign': campaign.name, 'contract': status['contract'], 'weights': status['reward_weights'],
              'stage_transitions': status.get('stage_transitions', []), 'recoveries': status.get('recoveries', []),
              'episodes': [], 'validation_errors': [], 'completed_steps': status['gameplay_training_steps']}
    started = time.perf_counter()
    for e in status['episodes']:
        number = e['episode']; prev = json.loads((campaign/f'initial-state-{number}.json').read_text())
        counts = Counter(); reward = defaultdict(float); first_hit = None; hit_times = []
        groups = defaultdict(lambda: {'n': 0, 'sum': 0., 'high': 0})
        power_sum = 0.; player_n = 0; seconds = 0.; lines = 0; last_action = None
        positions = Counter(); prediction_sum = Counter(); high_bomb_cells = Counter()
        with (campaign/f'episode-{number}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line); raw = row['raw']; t = row['telemetry']; policy = t['policy']
                lines += 1; seconds += 1/30
                for key, value in t['reward']['components'].items(): reward[key] += value
                for event in row['events']:
                    kind = event['kind']
                    if kind == 'hit':
                        hit_times.append(seconds)
                        if first_hit is None: first_hit = seconds
                    if kind == 'damage': counts['damage'] += event.get('amount', 0)
                    if kind == 'kill': counts['kills'] += 1
                    if kind == 'power': counts['power_event'] += 1
                if raw['stage'] == prev['stage'] and raw['stage_frame']-prev['stage_frame'] != 2:
                    counts['non_two_frame_transition'] += 1
                if policy['trained_updates'] != e['total_updates']-1:
                    counts['wrong_policy_update'] += 1
                if prev.get('player'):
                    player_n += 1; player = prev['player']; p = prev['power_raw']/20; power_sum += p
                    x, y = player['position']; positions[f'{int((x+192)//48)},{int(y//56)}'] += 1
                    counts['edge'] += abs(x) > 160
                    counts['bottom'] += y >= 380
                    counts['p0'] += p == 0; counts['p1'] += p >= 1; counts['p4'] += p >= 4
                    counts['invincible'] += player['invincibility_raw'] > 0
                    active = bool(prev.get('bomb') and prev['bomb']['state'] == 1)
                    group = 'active' if active else ('available' if p >= 1 else 'unavailable')
                    g=groups[group];g['n']+=1;g['sum']+=policy['bomb'];g['high']+=policy['bomb']>=.9
                    if group == 'unavailable' and policy['bomb'] >= .9:
                        high_bomb_cells[f'{int((x+192)//48)},{int(y//56)}'] += 1
                    counts['bomb_activations'] += not active and bool(raw.get('bomb') and raw['bomb']['state'] == 1)
                    action = policy['action'];counts['shoot']+=action[1];counts['focus']+=action[2]
                    counts['bomb_selected']+=action[3];counts['neutral']+=action[0]==0
                    counts['invalid_selected'] += group=='unavailable' and action[3]==1
                    counts[f'direction_{action[0]}'] += 1
                    if last_action is not None: counts['direction_switch'] += action[0]!=last_action[0]
                    last_action = action
                    for k in ('shoot','focus','bomb'): prediction_sum[k] += policy[k]
                    prediction_sum['direction_entropy'] += -sum(v*math.log(max(v,1e-30)) for v in policy['directions'])
                    if raw.get('player'):
                        q=raw['player']['position']
                        counts['no_displacement'] += math.dist([x,y],q) < 1e-5
                prev=raw
        if lines != e['steps'] or abs(sum(reward.values())-e['return']) > 1e-5 or len(hit_times)!=e['hits']:
            result['validation_errors'].append({'episode':number,'lines':lines,'expected':e['steps'],
                'reward_error':sum(reward.values())-e['return'],'hits':len(hit_times),'expected_hits':e['hits']})
        optimizer = {}
        progress = campaign/f'optimizer-{number}/progress.json'
        if progress.exists():
            optimizer=json.loads(progress.read_text().splitlines()[-1])
        result['episodes'].append({'episode':number,'steps':lines,'seconds':seconds,'first_hit':first_hit,
            'hit_times':hit_times,'hits':e['hits'],'return':e['return'],'terminated':e['terminated'],
            'recovery_update':e.get('recovery_update',False),'reward':dict(reward),'counts':dict(counts),
            'player_n':player_n,'power_sum':power_sum,'groups':dict(groups),'positions':dict(positions),
            'high_bomb_cells':dict(high_bomb_cells),'prediction_sum':dict(prediction_sum),'optimizer':optimizer})
        if number%25==0: print(f'{number}/{len(status["episodes"])} episodes ({time.perf_counter()-started:.1f}s)',flush=True)
    rows=result['episodes']; n=len(rows)
    spans=[(1,50),(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,n),(n-49,n),(n-99,n),(1,n)]
    result['windows']={f'{a}-{b}':summarize([r for r in rows if a<=r['episode']<=b]) for a,b in spans}
    rolling=[(i+1,i+50,mean(r['seconds'] for r in rows[i:i+50])) for i in range(n-49)]
    best=max(rolling,key=lambda x:x[2]);result['best_survival_window']={'start':best[0],'end':best[1],
        'metrics':summarize(rows[best[0]-1:best[1]])}
    result['top_survival']=[{'episode':r['episode'],'seconds':r['seconds'],'first_hit':r['first_hit'],'return':r['return']}
                            for r in sorted(rows,key=lambda r:r['seconds'],reverse=True)[:10]]
    completed={e['episode'] for e in status['episodes']}
    result['excluded_logs']=[p.name for p in campaign.glob('episode-*.jsonl') if int(p.stem.split('-')[-1]) not in completed]
    result['seconds_to_analyze']=time.perf_counter()-started
    (output/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'windows':result['windows'],'best_survival_window':result['best_survival_window'],
                      'errors':result['validation_errors'],'top_survival':result['top_survival']},indent=2))


def summarize(rows):
    n=len(rows);steps=sum(r['player_n'] for r in rows);seconds=sum(r['seconds'] for r in rows)
    counts=Counter()
    for r in rows:counts.update(r['counts'])
    keys=set().union(*(r['reward'] for r in rows));opt=set().union(*(r['optimizer'] for r in rows))
    return {'n':n,'seconds':mean(r['seconds'] for r in rows),'median_seconds':stats.median(r['seconds'] for r in rows),
        'first_hit':mean(r['first_hit'] for r in rows),'median_first_hit':stats.median(r['first_hit'] for r in rows if r['first_hit'] is not None),
        'first_hit_ge30_fraction':sum((r['first_hit'] or 0)>=30 for r in rows)/n,
        'return':mean(r['return'] for r in rows),'terminated':sum(r['terminated'] for r in rows),
        'damage_per_second':counts['damage']/seconds,'damage_per_play':counts['damage']/n,
        'kills_per_play':counts['kills']/n,'bombs_per_play':counts['bomb_activations']/n,
        'mean_power_time_weighted':sum(r['power_sum'] for r in rows)/steps,
        'fractions':{k:counts[k]/steps for k in ('p0','p1','p4','shoot','focus','neutral','edge','bottom','invincible','no_displacement','direction_switch','invalid_selected')},
        'reward':{k:sum(r['reward'].get(k,0) for r in rows)/n for k in sorted(keys)},
        'groups':{k:{'n':sum(r['groups'].get(k,{}).get('n',0) for r in rows),
                     'probability_mean':sum(r['groups'].get(k,{}).get('sum',0) for r in rows)/max(1,sum(r['groups'].get(k,{}).get('n',0) for r in rows)),
                     'high':sum(r['groups'].get(k,{}).get('high',0) for r in rows)} for k in ('available','unavailable','active')},
        'optimizer_means':{k:mean(r['optimizer'].get(k) for r in rows) for k in sorted(opt)},
        'direction_entropy':sum(r['prediction_sum'].get('direction_entropy',0) for r in rows)/steps,
        'directions':{str(i):counts[f'direction_{i}']/steps for i in range(9)},
        'frame_anomalies':counts['non_two_frame_transition'],'policy_anomalies':counts['wrong_policy_update']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();analyze(args.campaign,args.output)

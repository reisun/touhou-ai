"""Aggregate all paired seeds; retain per-seed effects and factorial interaction."""
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sweep_jitter_penalty import summarize as base_summary
OUT=pathlib.Path('artifacts/movement-history-20261001')
def summarize(rows):
    result=base_summary(rows)
    actions=[x['action'] for row in rows for x in row['trace']]
    result['fast_fraction']=sum(a[2]==0 for a in actions)/len(actions)
    result['standstill_fraction']=sum(a[0]==0 for a in actions)/len(actions)
    return result
summary={}
for penalty in (0.,.1):
    arms={}
    for history in (False,True):
        tasks={}
        for task in ('standard','challenge'):
            rows=[];seeds={}
            for seed in (7,17,27):
                d=OUT/f'seed-{seed}-history-{int(history)}-penalty-{penalty}'
                assert (d/'result.json').exists(),f'incomplete: {d}'
                part=json.loads((d/f'{task}.json').read_text());rows.extend(part)
                seeds[str(seed)]=summarize(part)
            tasks[task]=dict(pooled=summarize(rows),seeds=seeds)
        arms[str(int(history))]=tasks
    summary[str(penalty)]=arms
effects={}
for task in ('standard','challenge'):
    effects[task]={}
    for metric in ('survival','reversals','flags','movement_per_second'):
        differences={str(p):summary[str(p)]['1'][task]['pooled'][metric]-summary[str(p)]['0'][task]['pooled'][metric] for p in (0.,.1)}
        effects[task][metric]=dict(history_effect=differences,
            penalty_interaction=differences['0.1']-differences['0.0'])
result=dict(arms=summary,effects=effects,note='3 training seeds; pooled trials are not independent training replications; familiar evaluation seeds, not novel bullet families')
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

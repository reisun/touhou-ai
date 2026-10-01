import pathlib,json,numpy as np
root=pathlib.Path('artifacts/live-jitter-trial-20261001')
def summary(rows):
    seconds=sum(r['frames']/60 for r in rows)
    return dict(n=len(rows),survival=float(np.mean([r['success'] for r in rows])),seconds=float(np.mean([r['frames']/60 for r in rows])),
        p20_seconds=float(np.quantile([r['frames']/60 for r in rows],.2)),
        switches_per_second=sum(r['metrics']['action_switches'] for r in rows)/seconds,
        argmax_switches_per_second=sum(r['metrics']['argmax_switches'] for r in rows)/seconds,
        reversals_per_second=sum(r['metrics']['reversals'] for r in rows)/seconds,
        flags12_fraction=float(np.mean([r['metrics']['windows']['12']['fraction'] for r in rows])))
report={'system':{},'reward':{}}
for p in root.glob('system-*.json'):report['system'][p.stem]=summary(json.loads(p.read_text()))
for penalty in ('0.0','0.001','0.003'):
    arrays=[];individual={}
    for seed in (7,17,27):
        p=root/f'train-{seed}-{penalty}/evaluation.json'
        if p.exists():
            rows=json.loads(p.read_text());arrays+=rows;individual[str(seed)]=summary(rows)
    if arrays:report['reward'][penalty]=dict(pooled=summary(arrays),seeds=individual)
report['challenge_system']={p.stem:summary(json.loads(p.read_text())) for p in root.glob('challenge-system-*.json')}
report['challenge_reward']={}
for penalty in ('0.0','0.001','0.003'):
    arrays=[];individual={}
    for seed in (7,17,27):
        p=root/f'train-{seed}-{penalty}/challenge.json'
        if p.exists():
            rows=json.loads(p.read_text());arrays+=rows;individual[str(seed)]=summary(rows)
    if arrays:report['challenge_reward'][penalty]=dict(pooled=summary(arrays),seeds=individual)
(root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

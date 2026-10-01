import json,pathlib,numpy as np
root=pathlib.Path('artifacts/input-structure-jitter-20261001');summary={}
for mode in ('base','input_ema','feature_ema','blur_local'):
    rows=json.loads((root/f'trial-{mode}.json').read_text());groups={}
    for y in (160,400,'all'):
        a=[r for r in rows if y=='all' or r['start_y']==y];seconds=sum(r['frames']/60 for r in a)
        groups[str(y)]=dict(n=len(a),wins=sum(r['success'] for r in a),mean_seconds=seconds/len(a),
            argmax_switches_per_second=sum(r['metrics']['argmax_switches'] for r in a)/seconds,
            action_switches_per_second=sum(r['metrics']['action_switches'] for r in a)/seconds,
            reversals_per_second=sum(r['metrics']['reversals'] for r in a)/seconds,
            jitter_fraction=float(np.mean([r['metrics']['windows']['12']['fraction'] for r in a])))
    summary[mode]=groups
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

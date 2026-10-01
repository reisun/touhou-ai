import pathlib,json,numpy as np
root=pathlib.Path('artifacts/motion-jitter-20261001')
rows=[r for seed in (7,17,27) for r in json.loads((root/f'{seed}.json').read_text())]
summary={}
for mode in ('sample','greedy'):
    a=[r for r in rows if r['mode']==mode];seconds=sum(r['frames']/60 for r in a);steps=sum(len(r['trace']) for r in a)
    summary[mode]=dict(n=len(a),success=sum(r['success'] for r in a)/len(a),
        argmax_switches=sum(r['metrics']['argmax_switches'] for r in a),
        action_switches_per_second=sum(r['metrics']['action_switches'] for r in a)/seconds,
        reversals_per_second=sum(r['metrics']['reversals'] for r in a)/seconds,
        non_argmax_fraction=sum(r['metrics']['non_argmax'] for r in a)/steps,
        argmax_switch_episodes=sum(r['metrics']['argmax_switches']>0 for r in a),
        min_margin=min(x['margin'] for r in a for x in r['trace']))
    for success in (True,False):
        b=[r for r in a if r['success']==success]
        summary[mode][str(success)]={str(w):dict(episodes=len(b),episodes_flagged=sum(r['metrics']['windows'][str(w)]['flags']>0 for r in b),
            mean_episode_flag_fraction=float(np.mean([r['metrics']['windows'][str(w)]['fraction'] for r in b]))) for w in (6,12,24)}
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

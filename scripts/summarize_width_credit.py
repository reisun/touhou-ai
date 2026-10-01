import json,pathlib,statistics
root=pathlib.Path('artifacts/width-credit-20260930');report={}
for variant in ('control','width','credit','width64'):
    rows=[]
    for seed in (7,17,27):
        folder=(pathlib.Path('artifacts/critic-normalization-20260930/control') if variant=='control' else root/variant)/str(seed)
        if not (folder/'result.json').exists():continue
        r=json.loads((folder/'result.json').read_text())
        assert r['steps']==16384 and r['save_reload_equal']
        tail=json.loads((folder/'learning.json').read_text())[-8:]
        row=dict(seed=seed,sample=r['evaluations']['sample']['survival'],
            seconds=r['evaluations']['sample']['mean_frames']/60,
            greedy=r['evaluations']['greedy']['survival'],
            saturation=r['diagnostic']['last_tanh_saturation'],
            rmse=statistics.mean(x['rmse'] for x in tail),
            explained_variance=statistics.mean(x['explained_variance'] for x in tail))
        if variant!='control':
            row['frozen_greedy']=r['evaluations']['frozen_greedy']['survival']
            normal={e['seed']:e for e in r['evaluations']['greedy']['episodes']}
            row['frozen_matching_episodes']=sum(normal[e['seed']]['frames']==e['frames'] and normal[e['seed']]['success']==e['success'] for e in r['evaluations']['frozen_greedy']['episodes'])
        rows.append(row)
    if rows:report[variant]=dict(rows=rows,means={k:statistics.mean(r[k] for r in rows) for k in rows[0] if k!='seed'},
        sample_seed_sd=statistics.stdev(r['sample'] for r in rows) if len(rows)>1 else None)
(root/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

import json,pathlib,statistics
root=pathlib.Path('artifacts/width-credit-20260930/adam05')
report={}
for name,folder in [('control',pathlib.Path('artifacts/critic-normalization-20260930/control')),('adam05',root)]:
    rows=[]
    for seed in (7,17,27):
        r=json.loads((folder/str(seed)/'result.json').read_text())
        tail=json.loads((folder/str(seed)/'learning.json').read_text())[-8:]
        assert r['steps']==16384 and r['save_reload_equal']
        e=r['evaluations']
        row=dict(seed=seed,sample=e['sample']['survival'],greedy=e['greedy']['survival'],seconds=e['sample']['mean_frames']/60,
                 rmse=statistics.mean(x['rmse'] for x in tail),ev=statistics.mean(x['explained_variance'] for x in tail))
        if 'frozen_greedy' in e:
            row['frozen_greedy']=e['frozen_greedy']['survival']
            normal={a['seed']:a for a in e['greedy']['episodes']}
            row['frozen_matches']=sum(normal[a['seed']]['success']==a['success'] and normal[a['seed']]['frames']==a['frames'] for a in e['frozen_greedy']['episodes'])
        rows.append(row)
    report[name]=dict(rows=rows,means={k:statistics.mean(x[k] for x in rows) for k in rows[0] if k!='seed'})
(root/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

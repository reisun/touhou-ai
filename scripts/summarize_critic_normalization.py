import json,pathlib,statistics
ROOT=pathlib.Path('artifacts/critic-normalization-20260930')
summary={}
frozen_path=ROOT/'frozen-greedy.json'
frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
for variant in ('control','global','pre_tanh'):
    rows=[]
    for seed in (7,17,27):
        path=ROOT/variant/str(seed)
        if not (path/'result.json').exists():continue
        r=json.loads((path/'result.json').read_text())
        assert r['steps']==16384 and r['save_reload_equal']
        learning=json.loads((path/'learning.json').read_text())[-8:]
        key=f'{variant}/{seed}'
        if key in frozen:
            assert frozen[key]['n']==96
            normal={e['seed']:e for e in r['evaluations']['greedy']['episodes']}
            fixed=frozen[key]['episodes']
            frozen[key]['identical_success_and_frames']=sum(
                normal[e['seed']]['success']==e['success'] and normal[e['seed']]['frames']==e['frames'] for e in fixed)
        rows.append(dict(seed=seed,sample=r['evaluations']['sample']['survival'],
            greedy=r['evaluations']['greedy']['survival'],
            seconds=r['evaluations']['sample']['mean_frames']/60,
            saturation=r['diagnostic']['last_tanh_saturation'],
            training_seconds=r['training_seconds'],
            rmse=statistics.mean(x['rmse'] for x in learning),
            explained_variance=statistics.mean(x['explained_variance'] for x in learning)))
    if rows:
        summary[variant]=dict(rows=rows,means={k:statistics.mean(r[k] for r in rows) for k in rows[0] if k!='seed'})
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
if frozen:(ROOT/'frozen-greedy.json').write_text(json.dumps(frozen,indent=2))
print(json.dumps(summary,indent=2))

import json,pathlib,numpy as np
root=pathlib.Path('artifacts/spatial-inputs-20260929')
variants=('control','pixel1','action_grid','geometry');seeds=(7,17,27)
rows={v:[json.loads((root/v/str(s)/'result.json').read_text(encoding='utf-8')) for s in seeds] for v in variants}
for variant, group in rows.items():
    for seed, row in zip(seeds, group):
        assert row['variant']==variant and row['seed']==seed
        assert row['steps']==16384 and row['save_reload_equal']
        for mode in ('sample','greedy','frozen_sample','frozen_greedy'):
            evaluation=row['evaluations'][mode];episodes=evaluation['episodes']
            assert evaluation['n']==len(episodes)==96
            assert sorted(e['seed'] for e in episodes)==list(range(5000,5096))
            assert all(0<e['frames']<=600 and (not e['success'] or e['frames']==600) for e in episodes)
            assert np.isclose(evaluation['survival'],np.mean([e['success'] for e in episodes]))
            assert np.isclose(evaluation['mean_frames'],np.mean([e['frames'] for e in episodes]))
summary={}
for variant in variants:
    means={mode:{'survival':float(np.mean([r['evaluations'][mode]['survival'] for r in rows[variant]])),
                 'seconds':float(np.mean([r['evaluations'][mode]['mean_frames']/60 for r in rows[variant]]))}
           for mode in ('sample','greedy','frozen_sample','frozen_greedy')}
    differences=[];pairs=[]
    for control,row in zip(rows['control'],rows[variant]):
        a={e['seed']:e for e in control['evaluations']['sample']['episodes']}
        b={e['seed']:e for e in row['evaluations']['sample']['episodes']}
        assert set(a)==set(b)
        differences.append([int(b[k]['success'])-int(a[k]['success']) for k in sorted(a)])
        pairs.append(dict(seed=row['seed'],survival=row['evaluations']['sample']['survival'],seconds=row['evaluations']['sample']['mean_frames']/60,
            control_survival=control['evaluations']['sample']['survival'],control_seconds=control['evaluations']['sample']['mean_frames']/60))
    d=np.array(differences);rng=np.random.default_rng(2930);boot=[]
    for _ in range(10000):
        s=rng.integers(0,3,3);case=rng.integers(0,96,96);boot.append(d[s][:,case].mean())
    summary[variant]=dict(means=means,pairs=pairs,parameters=rows[variant][0]['parameters'],
        training_seconds_mean=float(np.mean([r['training_seconds'] for r in rows[variant]])),
        paired_mean_difference=float(d.mean()),paired_seed_case_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        improved_seeds=int(np.sum(d.mean(1)>0)),worse_seeds=int(np.sum(d.mean(1)<0)))
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

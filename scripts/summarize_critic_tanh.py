import json,pathlib,numpy as np
root=pathlib.Path('artifacts/critic-no-final-tanh-20260929')
controls=json.loads(pathlib.Path('artifacts/autumn-separate-clip-20260928/holdout.json').read_text(encoding='utf-8'))
paired=[];differences=[]
for seed in (7,17,27):
    before=next(r for r in controls if r['representation']=='cnn' and r['variant']=='separate' and r['seed']==seed)
    after=json.loads((root/str(seed)/'result.json').read_text(encoding='utf-8'))
    layer=json.loads((root/str(seed)/'layers.json').read_text(encoding='utf-8'))
    learning=json.loads((root/str(seed)/'learning.json').read_text(encoding='utf-8'))
    row=dict(seed=seed,before={k:{'survival':v['survival'],'seconds':v['mean_frames']/60} for k,v in before['evaluations'].items()},
        after={k:{'survival':v['survival'],'seconds':v['mean_frames']/60} for k,v in after['evaluations'].items()},
        value_range={k:v['value_range'] for k,v in layer.items()},
        critic_first_tanh_saturation={k:v['layers']['critic.mlp.1']['tanh_abs_above_099'] for k,v in layer.items()},
        last_eight_rollouts={key:float(np.mean([r[key] for r in learning[-8:]])) for key in ('value_std','target_std','explained_variance','rmse')})
    paired.append(row)
    b={r['seed']:r['success'] for r in before['evaluations']['sample']['episodes']}
    a={r['seed']:r['success'] for r in after['evaluations']['sample']['episodes']}
    assert set(a)==set(b)
    differences.append([int(a[k])-int(b[k]) for k in sorted(a)])
d=np.array(differences);rng=np.random.default_rng(928);boot=[]
for _ in range(10000):
    seeds=rng.integers(0,3,3);boot.append(np.mean([d[s,rng.integers(0,96,96)].mean() for s in seeds]))
summary=dict(paired=paired,mean={variant:{mode:{metric:float(np.mean([r[variant][mode][metric] for r in paired])) for metric in ('survival','seconds')} for mode in paired[0][variant]} for variant in ('before','after')},
    sample_difference=float(d.mean()),hierarchical_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
    caveat='Only three training seeds. Evaluation cases reused from earlier diagnostics; this is not a new untouched evaluation set.')
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

import pathlib,json,sys
import numpy as np

root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/cnn-update-credit-20260929')
records=[];updates=[]
for seed in (7,17,27):
    d=json.loads((root/f'{seed}.json').read_text(encoding='utf-8'))
    for u in d['updates']:
        rs=u['rows']; before=np.array([r['before'] for r in rs]); after=np.array([r['after'] for r in rs])
        updates.append(dict(seed=seed,update=u['update'],probes=len(rs),
            value_range=float(np.ptp([r['value'] for r in rs])),
            return_range=float(np.ptp([r['return_'] for r in rs])),
            max_action_probability_range=float(np.ptp(before,axis=0).max()),verification=u['verification']))
        for j,r in enumerate(rs):
            included=[b for b in u['batches'] if str(r['index']) in b['normalized_advantages']]
            excluded=[b for b in u['batches'] if str(r['index']) not in b['normalized_advantages']]
            norm=[b['normalized_advantages'][str(r['index'])] for b in included]
            rec=dict(seed=seed,update=u['update'],index=r['index'],frame=r['frame'],action=r['action'],death=r['death'],
                advantage=r['advantage'],value=r['value'],return_=r['return_'],normalized_advantages=norm,
                chosen_before=r['before'][r['action']],chosen_after=r['after'][r['action']],
                chosen_delta=float(after[j,r['action']]-before[j,r['action']]),
                included_delta=sum(b['chosen_delta'][j] for b in included),excluded_delta=sum(b['chosen_delta'][j] for b in excluded))
            for h in ('safe2','safe16'):
                safe=np.array(r[h]);rec[h+'_mixed']=bool(safe.any() and not safe.all());rec[h+'_chosen']=bool(safe[r['action']])
                rec[h+'_before']=float(before[j]@safe);rec[h+'_after']=float(after[j]@safe)
                rec[h+'_delta']=float((after[j]-before[j])@safe)
            assert abs(rec['chosen_delta']-rec['included_delta']-rec['excluded_delta'])<1e-6
            records.append(rec)

def group(rows):
    if not rows:return dict(n=0)
    return dict(n=len(rows),chosen_increased=sum(r['chosen_delta']>0 for r in rows),
        sign_changed_across_batches=sum(min(r['normalized_advantages'])<0<max(r['normalized_advantages']) for r in rows if r['normalized_advantages']),
        negative_signal_but_increased=sum(all(a<0 for a in r['normalized_advantages']) and r['chosen_delta']>0 for r in rows if r['normalized_advantages']),
        positive_signal_but_decreased=sum(all(a>0 for a in r['normalized_advantages']) and r['chosen_delta']<0 for r in rows if r['normalized_advantages']),
        all_normalized_negative=sum(all(a<0 for a in r['normalized_advantages']) for r in rows),
        any_normalized_positive=sum(any(a>0 for a in r['normalized_advantages']) for r in rows),
        mean_chosen_delta=float(np.mean([r['chosen_delta'] for r in rows])),
        safe2_mass_increased=sum(r['safe2_delta']>1e-7 for r in rows),
        safe16_mass_increased=sum(r['safe16_delta']>1e-7 for r in rows),
        mean_safe2_delta=float(np.mean([r['safe2_delta'] for r in rows])),
        mean_safe16_delta=float(np.mean([r['safe16_delta'] for r in rows])))
summary={'updates':updates,'all':group(records),'death':group([r for r in records if r['death']]),
    'by_seed':{str(s):group([r for r in records if r['death'] and r['seed']==s]) for s in (7,17,27)},
    'death_details':[r for r in records if r['death']]}
for h in ('safe2','safe16'):
    rows=[r for r in records if r[h+'_mixed']]
    summary[h+'_mixed']=group(rows)
    summary[h+'_mixed_chosen_safe']=group([r for r in rows if r[h+'_chosen']])
    summary[h+'_mixed_chosen_danger']=group([r for r in rows if not r[h+'_chosen']])
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(root/'scenes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('updates','death_details')},indent=2))

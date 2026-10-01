import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.audit_cnn_update_credit import run

root=pathlib.Path('artifacts/credit-v4-rollout-20260930');root.mkdir(exist_ok=True)
paired=[]
for seed in (7,17,27):
    path=root/f'{seed}.json'
    if not path.exists():run(seed,2,path,v4=True,rollout_norm=True)
    left=json.loads((pathlib.Path('artifacts/credit-v4-20260930')/f'{seed}.json').read_text())['updates'][0]
    right=json.loads(path.read_text())['updates'][0]
    assert left['steps']==right['steps'] and len(left['rows'])==len(right['rows'])
    assert right['verification']['plain_update_max_parameter_difference']==0
    for a,b in zip(left['rows'],right['rows']):
        for key in ('index','frame','action','death','advantage','value','return_','safe2','safe16','before'):
            assert a[key]==b[key],(seed,key)
        safe=np.asarray(a['safe2'],bool)
        paired.append(dict(seed=seed,index=a['index'],death=a['death'],mixed=bool(safe.any() and not safe.all()),
            control_chosen_delta=a['after'][a['action']]-a['before'][a['action']],
            credit_chosen_delta=b['after'][b['action']]-b['before'][b['action']],
            control_safe_delta=float((np.asarray(a['after'])-a['before'])@safe),
            credit_safe_delta=float((np.asarray(b['after'])-b['before'])@safe)))
(root/'paired-first-update.json').write_text(json.dumps(paired,indent=2))
for name,rows in [('all',paired),('mixed',[r for r in paired if r['mixed']]),('death',[r for r in paired if r['death']])]:
    print(name,len(rows),'safe delta',np.mean([r['control_safe_delta'] for r in rows]),np.mean([r['credit_safe_delta'] for r in rows]),flush=True)

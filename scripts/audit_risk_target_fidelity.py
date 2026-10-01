import sys,pathlib,json,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.autumn_ablation import NumericalAblation
from touhou_ai.risk_auxiliary import RiskTargets
from scripts.audit_observation_credit import snapshot,branch
scenes=[]
for ep in range(12):
    e=RiskTargets(lambda:NumericalAblation(relative=True));o,_=e.reset(seed=7000+ep);rng=np.random.default_rng(23000+ep)
    for step in range(300):
        y=o['risk_targets']
        if step%2==0 and 0<y.sum()<18 and e.unwrapped.frame<=584:scenes.append((snapshot(e.unwrapped),y))
        o,_,done,_,_=e.step([int(rng.integers(9)),0,int(rng.integers(2)),0])
        if done:break
indices=np.random.default_rng(452).choice(len(scenes),min(32,len(scenes)),replace=False)
actual=[];predicted=[]
for index in indices:
    state,y=scenes[index];_,safe=branch(state);actual.extend((~safe).tolist());predicted.extend(y.astype(bool).tolist())
a=np.array(actual);p=np.array(predicted)
r={'eligible_mixed_scenes':len(scenes),'sampled_scenes':len(indices),'action_scenes':len(a),'agreement':float(np.mean(a==p)),'predicted_safe_but_hit':int(np.sum(~p&a)),'predicted_hit_but_safe':int(np.sum(p&~a)),'scope':'random subset of held-out mixed-risk probe scenes; simulator chosen2F then stationary14F; diagnostic only; no future branching used in training'}
pathlib.Path('artifacts/autumn-risk-aux-20260928/target-fidelity.json').write_text(json.dumps(r,indent=2));print(r)

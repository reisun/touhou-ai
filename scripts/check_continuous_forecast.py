"""Validate the new path approximation against real simulator branches."""
import sys,pathlib,copy,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.validate_spatial_inputs import scenes
from touhou_ai.risk_refinements import forecast
from touhou_ai.spatial_input_candidates import TILES
from touhou_ai.autumn_ablation import GridAblation

records=[]
for raw,state in scenes():
    if state['frame']>584:continue
    prediction=forecast(raw,continuous=True);truth=np.zeros_like(prediction)
    for direction,(iy,ix) in enumerate(TILES):
        for focus in (0,1):
            e=GridAblation()
            for k,v in state.items():setattr(e,k,copy.deepcopy(v))
            e.observe=lambda:{}
            for step in range(8):
                _,_,done,_,_=e.step([direction,0,focus,0])
                if step==0:truth[focus,iy,ix]=e.dead
                if done:break
            truth[2+focus,iy,ix]=e.dead
    records.append(dict(frame=state['frame'],predicted=prediction.tolist(),actual=truth.tolist()))
summary={}
for name,sl in [('2f',slice(0,2)),('16f',slice(2,4))]:
    p=np.array([r['predicted'] for r in records])[:,sl].astype(bool)
    a=np.array([r['actual'] for r in records])[:,sl].astype(bool)
    summary[name]=dict(n=int(p.size),agreement=float((p==a).mean()),false_safe=int((~p&a).sum()),false_danger=int((p&~a).sum()))
out=pathlib.Path('artifacts/risk-refinements-20260929/continuous-fidelity.json')
out.write_text(json.dumps(dict(summary=summary,scenes=records),indent=2))
print(json.dumps(summary,indent=2))

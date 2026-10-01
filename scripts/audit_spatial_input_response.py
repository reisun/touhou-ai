"""Read-only reaction of the 12 trained policies to input interventions."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.validate_spatial_inputs import scenes
from scripts.compare_spatial_inputs import OUT,SEEDS
from scripts.audit_cnn_layers import outputs
from touhou_ai.spatial_input_candidates import CandidateEncoder,TILES
from touhou_ai.dual_grid import DualGridContract
from touhou_ai.separate_clip_ppo import SeparateClipPPO

torch.set_num_threads(1)
raws=[raw for raw,_ in scenes()]
valid=[i for i,raw in enumerate(raws) if raw['stage_frame']<=584]
records=json.loads((OUT/'predictor-fidelity.json').read_text(encoding='utf-8'))['scenes']
assert len(valid)==len(records)==59
assert [raws[i]['stage_frame'] for i in valid]==[r['frame'] for r in records]
truth={name:np.array([[[r['actual'][offset+focus][iy][ix] for focus in (0,1)] for iy,ix in TILES] for r in records]).reshape(-1,18)
       for name,offset in [('2f',0),('16f',2)]}
def collision_probability(p):
    result={}
    for name,labels in truth.items():
        mixed=(labels.sum(1)>0)&(labels.sum(1)<18)
        risk=(p[valid]*labels).sum(1)
        result[name]=dict(mixed_scenes=int(mixed.sum()),mean_probability=float(risk.mean()),mixed_mean_probability=float(risk[mixed].mean()))
    return result
result={}
for variant in ('control','pixel1','action_grid','geometry'):
    encoder=DualGridContract() if variant=='control' else CandidateEncoder(variant)
    obs=[encoder.encode(raw) for raw in raws]
    batch={k:np.stack([o[k] for o in obs]) for k in obs[0]}
    result[variant]=[]
    for seed in SEEDS:
        model=SeparateClipPPO.load(OUT/variant/str(seed)/'model.zip',device='cpu')
        p,v=outputs(model.policy,batch)
        def compare(other):
            q,w=outputs(model.policy,other)
            tv=np.abs(p-q).sum(1)/2
            return dict(mean_policy_tv=float(tv.mean()),max_policy_tv=float(tv.max()),
                        mean_value_change=float(np.abs(v-w).mean()),actual_collision_probability=collision_probability(q))
        other={k:(np.roll(x,len(x)//2,axis=0) if k.endswith('_grid') else x) for k,x in batch.items()}
        row=dict(seed=seed,grid_permutation=compare(other),value_range=float(np.ptp(v)),actual_collision_probability=collision_probability(p))
        if variant in ('action_grid','geometry'):
            other={k:x.copy() for k,x in batch.items()}
            if variant=='action_grid':other['action_grid'].fill(0)
            else:other['local_grid'][:,6:]=0
            row['added_input_zero']=compare(other)
            # Both of these connecting weight blocks were initialized to zero.
            extractor=model.policy.pi_features_extractor
            weight=extractor.merge[0].weight[:,321:] if variant=='action_grid' else extractor.local[0].weight[:,6:]
            row['added_actor_weight_l2']=float(torch.linalg.vector_norm(weight).detach())
        result[variant].append(row)
        print(variant,seed,row,flush=True)
(OUT/'input-response.json').write_text(json.dumps(dict(scenes=len(raws),
    caveat='Counterfactual input interventions measure reaction, not accuracy or causal survival improvement. Joint probabilities are products of the independent 9+2 heads, not an 18-output policy.',
    results=result),indent=2),encoding='utf-8')

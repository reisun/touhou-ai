"""Observable predictor fidelity against simulator branches; benchmark separately."""
import sys,pathlib,json,copy,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.spatial_input_candidates import CandidateEnv,CandidateEncoder,risk_grid,TILES,VARIANTS
from touhou_ai.dual_grid import DualGridContract
from touhou_ai.autumn_ablation import GridAblation
from scripts.audit_cnn_update_credit import FIELDS
from scripts.compare_spatial_inputs import OUT,build
from touhou_ai.simulation_speed import distributions

def scenes():
    selected=[];rng=np.random.default_rng(740)
    for seed in range(8):
        e=CandidateEnv('action_grid');e.observe=lambda:{};e.reset(seed=7400+seed);history=[]
        for t in range(300):
            state={k:copy.deepcopy(getattr(e,k)) for k in FIELDS}
            history.append((e.raw_observation(),state))
            _,_,done,_,_=e.step([int(rng.integers(9)),0,int(rng.integers(2)),0])
            if done:break
        for i in sorted(set([0,len(history)//3,2*len(history)//3]+list(range(max(0,len(history)-5),len(history))))):selected.append(history[i])
    return selected

def fidelity(data):
    records=[]
    for raw,state in data:
        if state['frame']>584:continue # Keep the full 16F branch inside the episode horizon.
        predicted=risk_grid(raw);truth=np.zeros_like(predicted)
        for direction,(iy,ix) in enumerate(TILES):
            for focus in (0,1):
                e=GridAblation()
                for k,v in state.items():setattr(e,k,copy.deepcopy(v))
                e.observe=lambda:{}
                _,_,done,_,_=e.step([direction,0,focus,0]);truth[focus,iy,ix]=e.dead
                for _ in range(7):
                    if done:break
                    _,_,done,_,_=e.step([0,0,focus,0])
                truth[2+focus,iy,ix]=e.dead
        records.append(dict(frame=int(state['frame']),predicted=predicted.tolist(),actual=truth.tolist()))
    result={}
    for name,sl in [('2f',slice(0,2)),('16f',slice(2,4))]:
        p=np.array([r['predicted'] for r in records])[:,sl].astype(bool)
        a=np.array([r['actual'] for r in records])[:,sl].astype(bool)
        result[name]=dict(n=int(p.size),agreement=float((p==a).mean()),false_safe=int((~p&a).sum()),false_danger=int((p&~a).sum()),actual_hits=int(a.sum()))
    (OUT/'predictor-fidelity.json').write_text(json.dumps(dict(summary=result,scenes=records),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

def benchmark(data):
    torch.set_num_threads(1);raws=[r for r,_ in data];encoders={'control':DualGridContract(),**{v:CandidateEncoder(v) for v in VARIANTS}}
    timing={v:[] for v in encoders};observations={v:[] for v in encoders};rng=np.random.default_rng(919)
    for repeat in range(3):
        for raw in raws:
            for variant in rng.permutation(list(encoders)):
                start=time.perf_counter();obs=encoders[variant].encode(raw);timing[variant].append((time.perf_counter()-start)*1000)
                if repeat==0:observations[variant].append(obs)
    result={}
    for variant in encoders:
        m,_=build(variant,7);times=[]
        with torch.inference_mode():
            for repeat in range(3):
                for obs in observations[variant]:
                    start=time.perf_counter();distributions(m.policy,obs);times.append((time.perf_counter()-start)*1000)
        sample=observations[variant][0]
        storage=sum(x.size*(2 if k.endswith('_grid') else 4) for k,x in sample.items())
        result[variant]=dict(encode_median_ms=float(np.median(timing[variant])),encode_p95_ms=float(np.quantile(timing[variant],.95)),
            actor_median_ms=float(np.median(times)),actor_p95_ms=float(np.quantile(times,.95)),rollout_observation_bytes=storage,
            rollout512_observation_mib=storage*512/2**20,parameters=sum(x.numel() for x in m.policy.parameters()),shapes={k:list(x.shape) for k,x in sample.items()})
    (OUT/'benchmark.json').write_text(json.dumps(dict(scenes=len(raws),repeats=3,threads=1,results=result),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    data=scenes()
    if '--benchmark' in sys.argv:benchmark(data)
    else:fidelity(data)

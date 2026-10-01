"""Paired training-initialization and common-evaluation-case bootstrap."""
import sys,pathlib,json,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.replicate_critic_stability import OUT,FRESH,OLD

def describe(a):
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),median=float(np.median(a)),below10=int((a<.1).sum()))

def bootstrap(a,b,draws=20000):
    rng=np.random.default_rng(292609);k,n=a.shape;ratios=[];diff=[]
    for _ in range(draws):
        seeds=rng.integers(k,size=k);cases=rng.integers(n,size=n)
        x=a[seeds][:,cases].mean(1);y=b[seeds][:,cases].mean(1)
        sx=x.std(ddof=1);sy=y.std(ddof=1)
        ratios.append(sy/sx if np.ptp(x)>0 else (1 if np.ptp(y)==0 else np.inf));diff.append(y.mean()-x.mean())
    return dict(draws=draws,sd_ratio_95=np.quantile(ratios,[.025,.975]).tolist(),mean_difference_95=np.quantile(diff,[.025,.975]).tolist())

def summarize(seeds):
    arrays={v:[] for v in ('control','no_tanh')};times={v:[] for v in arrays};pairs=[]
    for seed in seeds:
        pair={'seed':seed}
        for variant in arrays:
            row=json.loads((OUT/str(seed)/variant/'result.json').read_text(encoding='utf-8'))
            e=sorted(row['episodes'],key=lambda r:r['seed'])
            assert [r['seed'] for r in e]==list(range(5000,5192))
            arrays[variant].append([r['success'] for r in e]);times[variant].append([r['frames']/60 for r in e])
            pair[variant]=dict(survival=row['survival'],seconds=row['seconds_alive'])
        pairs.append(pair)
    a,b=[np.array(arrays[v],float) for v in arrays]
    report=dict(seeds=seeds,pairs=pairs,control=describe(a.mean(1)),no_tanh=describe(b.mean(1)),
        sd_ratio=float(b.mean(1).std(ddof=1)/a.mean(1).std(ddof=1)),mean_difference=float(b.mean()-a.mean()),
        bootstrap=bootstrap(a,b),panels={},duration={v:describe(np.array(t).mean(1)) for v,t in times.items()})
    for v in report['duration'].values():v.pop('below10')
    for offset in (0,96):report['panels'][offset]=dict(control=describe(a[:,offset:offset+96].mean(1)),no_tanh=describe(b[:,offset:offset+96].mean(1)))
    return report

if __name__=='__main__':
    result=dict(primary_additional=summarize(FRESH),secondary_all=summarize(OLD+FRESH),original_three=summarize(OLD))
    ci=result['primary_additional']['bootstrap']
    result['stability_confirmed_by_prespecified_criterion']=ci['sd_ratio_95'][1]<1
    result['mean_noninferiority_passed']=ci['mean_difference_95'][0]>-.02
    result['adoption_criterion_passed']=result['stability_confirmed_by_prespecified_criterion'] and result['mean_noninferiority_passed']
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

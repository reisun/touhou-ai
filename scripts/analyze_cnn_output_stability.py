"""Seed-paired, evaluation-case-paired bootstrap for the fixed CNN replication."""
import json,numpy as np
from scripts.replicate_cnn_output_stability import OUT,FRESH,OLD

def metrics(a,b):
 sa=float(a.std(ddof=1)) if np.ptp(a)>0 else 0.;sb=float(b.std(ddof=1)) if np.ptp(b)>0 else 0.
 ratio=sb/sa if sa>0 else (1. if sb==0 else float('inf'))
 return [ratio,float(b.mean()-a.mean()),float(b.std(ddof=1)-a.std(ddof=1))]

def bootstrap(a,b,draws=20000,resample_cases=True):
 rng=np.random.default_rng(20260929);values=[];k,n=a.shape
 for _ in range(draws):
  seeds=rng.integers(k,size=k);cases=rng.integers(n,size=n) if resample_cases else np.arange(n)
  x=a[seeds][:,cases].mean(1);y=b[seeds][:,cases].mean(1)
  values.append(metrics(x,y))
 values=np.asarray(values)
 return {'draws':draws,'method':'paired training seeds and shared evaluation cases' if resample_cases else 'paired training seeds, fixed evaluation cases','sd_ratio_joint_over_independent_95':np.quantile(values[:,0],[.025,.975]).tolist(),'mean_difference_95':np.quantile(values[:,1],[.025,.975]).tolist(),'sd_difference_95':np.quantile(values[:,2],[.025,.975]).tolist()}

def describe(x):
 return {'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'min':float(x.min()),'max':float(x.max()),'median':float(np.median(x)),'iqr':float(np.quantile(x,.75)-np.quantile(x,.25)),'below_10_percent':int((x<.1).sum())}

def summarize(seeds):
 data={v:[] for v in ['independent','joint']};duration={v:[] for v in data};pairs=[]
 for seed in seeds:
  pair={'seed':seed}
  for variant in data:
   r=json.loads((OUT/str(seed)/variant/'result.json').read_text());episodes=sorted(r['episodes'],key=lambda e:e['seed'])
   assert [e['seed'] for e in episodes]==list(range(5000,5192))
   data[variant].append([e['success'] for e in episodes]);duration[variant].append([e['frames']/60 for e in episodes]);pair[variant]={'survival':r['survival'],'seconds':r['seconds_alive']}
  pairs.append(pair)
 a,b=[np.asarray(data[v],dtype=float) for v in data];ta,tb=[np.asarray(duration[v],dtype=float) for v in data]
 result={'seeds':seeds,'n_training_pairs':len(seeds),'episodes_per_model':192,'pairs':pairs,'independent':describe(a.mean(1)),'joint':describe(b.mean(1)),'observed_sd_ratio':metrics(a.mean(1),b.mean(1))[0],'mean_difference':float(b.mean()-a.mean()),'two_axis_bootstrap':bootstrap(a,b),'seed_only_bootstrap':bootstrap(a,b,resample_cases=False),'duration':{'independent':describe(ta.mean(1)),'joint':describe(tb.mean(1))},'panels':{}}
 for d in result['duration'].values():d.pop('below_10_percent',None)
 for start in [0,96]:
  x=a[:,start:start+96].mean(1);y=b[:,start:start+96].mean(1)
  result['panels'][str(start)]={'independent':describe(x),'joint':describe(y),'sd_ratio':metrics(x,y)[0]}
 return result

def main():
 missing=[(s,v) for s in FRESH+OLD for v in ['independent','joint'] if not (OUT/str(s)/v/'result.json').exists()]
 if missing:raise SystemExit('Incomplete: '+str(missing))
 report={'primary_fresh':summarize(FRESH),'secondary_all':summarize(OLD+FRESH),'exploratory_original':summarize(OLD)}
 p=report['primary_fresh']['two_axis_bootstrap'];report['stability_criterion_passed']=p['sd_ratio_joint_over_independent_95'][1]<1;report['mean_noninferiority_criterion_passed']=p['mean_difference_95'][0]>-.02;report['adoption_criterion_passed']=report['stability_criterion_passed'] and report['mean_noninferiority_criterion_passed']
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{q:v for q,v in r.items() if q!='pairs'} if isinstance(r,dict) else r for k,r in report.items()},indent=2))
if __name__=='__main__':main()

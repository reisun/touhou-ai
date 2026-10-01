"""Bounded increasing-penalty comparison; stops on replicated jitter reduction."""
import sys,pathlib,json,subprocess,time,concurrent.futures
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/live-jitter-trial-20261001'
def summarize(rows):
    secs=sum(r['frames']/60 for r in rows)
    return dict(n=len(rows),survival=float(np.mean([r['success'] for r in rows])),seconds=secs/len(rows),
        reversals=sum(r['metrics']['reversals'] for r in rows)/secs,
        flags=float(np.mean([r['metrics']['windows']['12']['fraction'] for r in rows])),
        movement_per_second=sum(np.linalg.norm(x['delta']) for r in rows for x in r['trace'])/secs)
def assess(p):
    result={}
    for file in ('evaluation.json','challenge.json'):
        arrays=[];seeds={}
        for seed in (7,17,27):
            rows=json.loads((OUT/f'train-{seed}-{p}'/file).read_text());arrays+=rows;seeds[str(seed)]=summarize(rows)
        result[file]=dict(pooled=summarize(arrays),seeds=seeds)
    return result
def worker(seed,p):
    log=OUT/f'sweep-{seed}-{p}.log'
    with log.open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/live_jitter_trial.py'),'sweep',str(seed),str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
def main():
    report=dict(state='running',levels=[.01,.03,.1,.3,1.],criterion='Both reversal rate and 12F window fraction <=80% control in both tasks; at least 2/3 seeds improve both metrics in each task. Performance reported separately.',baseline=assess(0.0),results={})
    def save():
        temp=OUT/'sweep-status.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(OUT/'sweep-status.json')
    try:
        for p in report['levels']:
            report['current_penalty']=p;save();print('START',p,flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda seed:worker(seed,p),(7,17,27)))
            result=assess(p);passed=True
            for task,v in result.items():
                b=report['baseline'][task]
                passed &= all(v['pooled'][metric]<=.8*b['pooled'][metric] for metric in ('reversals','flags'))
                passed &= sum(all(v['seeds'][s][metric]<b['seeds'][s][metric] for metric in ('reversals','flags')) for s in v['seeds'])>=2
            report['results'][str(p)]=dict(result=result,jitter_reduced=bool(passed));save();print('RESULT',p,json.dumps(report['results'][str(p)]),flush=True)
            if passed:report['state']='complete_reduced';break
        else:report['state']='complete_limit'
        save()
    except BaseException as e:
        report['state']='failed';report['error']=repr(e);save();raise
if __name__=='__main__':main()

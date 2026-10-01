"""Bounded two-worker orchestration; does not control or modify real training."""
import pathlib,sys,json,subprocess,concurrent.futures,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/movement-history-20261001'
OUT.mkdir(exist_ok=True)
jobs=[(s,h,p) for s in (7,17,27) for p in (.1,0.) for h in (False,True)]
(OUT/'manifest.json').write_text(json.dumps(dict(seeds=[7,17,27],steps=8192,
    history='last 6 chosen direction vectors, focus, valid bit; oldest first; cleared at reset',
    comparison='2x2 history/zero-history and jitter penalty 0/0.1; identical 24-input extension capacity and initial outputs',
    source=json.loads((ROOT/'artifacts/live-jitter-trial-20261001/manifest.json').read_text()),
    evaluation='64 paired seeds5000..5063/RNG19000..19063, standard and y160; 600F Autumn Sky approximation',
    live_modified=False),indent=2))
def worker(job):
    s,h,p=job;name=f'seed-{s}-history-{int(h)}-penalty-{p}';dest=OUT/name
    if (dest/'result.json').exists():return name
    with (OUT/f'{name}.log').open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/compare_movement_history.py'),str(s),str(int(h)),str(p)],
            cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return name
state=dict(state='running',completed=[],total=len(jobs),started=time.time())
def save():
    p=OUT/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'status.json')
save()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(worker,j) for j in jobs]):
            state['completed'].append(future.result());save();print(state['completed'][-1],flush=True)
    state['state']='complete';state['seconds']=time.time()-state['started'];save()
except BaseException as e:
    state.update(state='failed',error=repr(e));save();raise

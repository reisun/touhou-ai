import pathlib,sys,json,time,subprocess,concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/two-frame-plan-20261002'
jobs=[(7,False,True),(7,True,True)]+[(seed,split,False) for seed in (7,17,27) for split in (False,True)]
state=dict(state='running',completed=[],total=len(jobs),started=time.time())
def save():
 p=OUT/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'status.json')
def run(job):
 seed,split,initial=job;name=f'{seed}-{int(split)}'+('-initial' if initial else '')
 result=OUT/f'initial-{int(split)}.json' if initial else OUT/f'{seed}-{int(split)}/result.json'
 if result.exists():return name
 command=[sys.executable,str(ROOT/'scripts/compare_two_frame_plan.py'),str(seed),str(int(split))]+(['--initial'] if initial else [])
 with (OUT/f'{name}.log').open('w') as log:
  subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
 return name
save()
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
   state['completed'].append(future.result());save();print(state['completed'][-1],flush=True)
 state.update(state='complete',seconds=time.time()-state['started']);save()
except BaseException as exc:state.update(state='failed',error=repr(exc));save();raise

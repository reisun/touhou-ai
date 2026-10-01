"""Finish authorized sequential offline experiments; no live controls."""
import pathlib,subprocess,sys,time,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/risk-refinements-20260929'
def complete(variant):
    return all((OUT/variant/str(s)/'result.json').exists() for s in (7,17,27))
deadline=time.monotonic()+10800
while not (complete('control') and complete('urgency')):
    if time.monotonic()>deadline:raise TimeoutError('Stage 1 did not finish')
    time.sleep(5)
for variant in ('continuous','scale'):
    subprocess.run([sys.executable,'scripts/report_risk_refinements.py'],cwd=ROOT,stdout=(OUT/f'before-{variant}-summary.log').open('w'),check=True)
    print('STAGE START',variant,flush=True)
    jobs=[]
    for seed in (7,17,27):
        if (OUT/variant/str(seed)/'result.json').exists():continue
        log=(OUT/f'{variant}-{seed}.log').open('w')
        p=subprocess.Popen([sys.executable,'scripts/compare_risk_refinements.py',variant,str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,seed))
    while any(p.poll() is None for p,_,_ in jobs):
        time.sleep(5)
    for p,log,seed in jobs:
        log.close()
        if p.returncode:raise RuntimeError(f'{variant}/{seed} failed; inspect log')
    assert complete(variant)
    print('STAGE COMPLETE',variant,flush=True)
subprocess.run([sys.executable,'scripts/report_risk_refinements.py'],cwd=ROOT,stdout=(OUT/'final-summary.log').open('w'),check=True)
print('ALL COMPLETE',flush=True)

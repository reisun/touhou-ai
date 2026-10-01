"""Preserve the current full-policy checkpoint at its next update boundary."""
import pathlib,json,time,hashlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
record=json.loads((root/'.runtime/live-learning.json').read_text())
expected=sys.argv[1] if len(sys.argv)>1 else 'live-learning-20260930-023754-eae16e'
assert record['RunId']==expected
folder=pathlib.Path(record['Output']);deadline=time.monotonic()+900
while time.monotonic()<deadline:
    s=json.loads((folder/'status.json').read_text())
    if s['status']=='stopped':break
    if s['status']=='failed':raise RuntimeError(s.get('error'))
    if s['phase'] in ('optimizing_at_game_over','optimizing_recovery'):
        (folder/'STOP').touch()
    time.sleep(.2)
else:raise TimeoutError('No update boundary')
e=s['episodes'][-1];checkpoint=folder/e['checkpoint']
assert e['reload_verified'] and e['reload_rng_verified']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==e['checkpoint_sha256']
out=root/'artifacts/stage-transition-fix-20260930';out.mkdir(exist_ok=True)
name='source.json' if len(sys.argv)==1 else f'source-{expected}.json'
(out/name).write_text(json.dumps(dict(record=record,checkpoint=str(checkpoint),status=s),indent=2))
print(checkpoint,flush=True)

"""Save and stop only the diagnosed learner, at the game-over update boundary."""
import json,pathlib,time,hashlib
root=pathlib.Path(__file__).resolve().parents[1]
out=root/'artifacts/action-grid-regression-20260929'
run=json.loads((root/'.runtime/live-learning.json').read_text())
assert run['RunId']=='live-learning-20260929-175742-f21a7b'
folder=pathlib.Path(run['Output']);deadline=time.monotonic()+180;requested=False
while time.monotonic()<deadline:
    s=json.loads((folder/'status.json').read_text())
    if s['status']=='stopped':break
    if s['status']=='failed':raise RuntimeError(s.get('error'))
    if s['phase']=='optimizing_at_game_over' and not requested:
        (folder/'STOP').touch();requested=True;print('STOP requested at boundary',s['updates']+1,flush=True)
    time.sleep(.2)
else:raise TimeoutError('No clean boundary stop')
e=s['episodes'][-1]
assert e['reload_verified'] and not e['truncated'] and e['updated_at_game_over']
assert hashlib.sha256((folder/e['checkpoint']).read_bytes()).hexdigest()==e['checkpoint_sha256']
(out/'stopped-run.json').write_text(json.dumps(dict(run=run,checkpoint=str(folder/e['checkpoint']),status=s),indent=2))
print(json.dumps(dict(updates=s['updates'],steps=s['gameplay_training_steps'],checkpoint=e['checkpoint'])))

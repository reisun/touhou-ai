"""Stop the explicitly selected evasion run after saving a complete game."""
import json, pathlib, time, hashlib
root=pathlib.Path(__file__).resolve().parents[1]
record=json.loads((root/'.runtime/live-learning.json').read_text(encoding='utf-8-sig'))
assert record['RunId']=='live-learning-20260929-221149-9bf836'
folder=pathlib.Path(record['Output'])
out=root/'artifacts/full-transfer-20260930';out.mkdir(exist_ok=True)
deadline=time.monotonic()+900
while time.monotonic()<deadline:
    s=json.loads((folder/'status.json').read_text())
    if s['status']=='failed':raise RuntimeError(s.get('error'))
    if s['status']=='stopped':break
    if s['phase']=='optimizing_at_game_over':(folder/'STOP').touch()
    time.sleep(.2)
else:raise TimeoutError('No completed boundary')
e=s['episodes'][-1]
assert e['reload_verified'] and not e['truncated'] and e['updated_at_game_over']
checkpoint=folder/e['checkpoint']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==e['checkpoint_sha256']
(out/'source.json').write_text(json.dumps(dict(record=record,checkpoint=str(checkpoint),status=s),indent=2))
print(checkpoint,flush=True)

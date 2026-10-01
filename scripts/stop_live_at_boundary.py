"""Request STOP only while an episode is already being saved at game over."""
import pathlib,json,re,time,hashlib
root=pathlib.Path(__file__).resolve().parents[1]
out=root/'artifacts/live-action-grid-cutover-20260929';out.mkdir(exist_ok=True)
record=(root/'.runtime/live-learning.json').read_text(encoding='utf-8-sig');run=json.loads(record)['RunId']
assert re.fullmatch(r'live-learning-\d{8}-\d{6}-[a-f0-9]{6}',run)
assert run=='live-learning-20260929-071910-9a4a78','Refuse to stop an unexpected learner'
folder=root/'artifacts'/run
for name,content in [('previous-run.json',record),('previous-profile.json',(root/'configs/sharu-inspired-v1.json').read_text(encoding='utf-8'))]:
    target=out/name
    if target.exists():assert target.read_text(encoding='utf-8')==content
    else:target.write_text(content,encoding='utf-8')
deadline=time.monotonic()+600;stamp=None;requested=False
while time.monotonic()<deadline:
    path=folder/'status.json';current=path.stat().st_mtime_ns
    if current==stamp:time.sleep(.2);continue
    stamp=current;status=json.loads(path.read_text(encoding='utf-8'))
    if status['status']=='failed':raise RuntimeError(status.get('error','learner failed'))
    if status['status']=='stopped':break
    if status.get('phase')=='optimizing_at_game_over' and not requested:
        (folder/'STOP').touch();requested=True
        print('Requested boundary stop after update',status['updates']+1,flush=True)
    time.sleep(.2)
else:raise TimeoutError('No completed boundary stop within 600 seconds')
last=status['episodes'][-1]
assert last['reload_verified'] and last['updated_at_game_over'] and not last['truncated']
assert hashlib.sha256((folder/last['checkpoint']).read_bytes()).hexdigest()==last['checkpoint_sha256']
(out/'previous-final-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(dict(run=run,updates=status['updates'],steps=status['gameplay_training_steps'],checkpoint=last['checkpoint'],sha256=last['checkpoint_sha256'],phase=status['phase'])),flush=True)

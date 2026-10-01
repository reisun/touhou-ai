"""Bounded passive FPS sampling and sparse read-only observations; no control."""
import json,pathlib,time,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.telemetry_memory import read
from touhou_ai.live_inspect import verify_game
from touhou_ai.windows_probe import ReadOnlyProcess
from touhou_ai.th10_reader import Th10Reader
root=pathlib.Path(__file__).resolve().parents[1]
out=root/'artifacts/fps-diagnosis-20260930';out.mkdir(exist_ok=True)
record=json.loads((root/'.runtime/live-learning.json').read_text());folder=pathlib.Path(record['Output'])
(out/'run.json').write_text(json.dumps(record))
start=time.monotonic();previous=None;rows=[];raws=[];next_raw=0;proc=None;pid=None
try:
 while time.monotonic()-start<240:
  item=read(previous=previous)
  if item:
   previous,payload=item;t=json.loads(payload);learning=t.get('learning') or {};g=t.get('game') or {}
   if learning.get('phase')=='playing' and t.get('episode_id','').startswith(record['RunId']):
    entities=t.get('entities') or {};bs=entities.get('bullets') or [];en=entities.get('enemies') or []
    frame=t.get('game_frame',0);phase='boss' if g.get('stage')==1 and frame>=4000 and any(e.get('is_boss') for e in en) else 'road' if g.get('stage')==1 and frame<2400 else 'other'
    rows.append(dict(time=t['timestamp'],episode=t['episode_id'],stage=g.get('stage'),frame=frame,steps=learning['steps'],phase=phase,bullets=len(bs),enemies=len(en),sample_ms=(t.get('metrics') or {}).get('sample_ms')))
    if time.monotonic()>=next_raw and len(raws)<40:
     next_raw=time.monotonic()+6
     try:
      status=json.loads((folder/'status.json').read_text());newpid=status['game_pid']
      if pid!=newpid:
       if proc:proc.close()
       config=verify_game(newpid);proc=ReadOnlyProcess(newpid,config['executable']);reader=Th10Reader(proc);pid=newpid
      before=reader.globals();raw=reader.snapshot();after=reader.globals()
      if before['stage']==after['stage']==raw['stage'] and before['stage_frame']==after['stage_frame']==raw['stage_frame'] and raw['player']:
       raws.append(dict(phase=phase,raw=raw))
     except (ValueError,OSError):pass
  time.sleep(.05)
finally:
 if proc:proc.close()
 (out/'telemetry.json').write_text(json.dumps(rows))
 (out/'raws.json').write_text(json.dumps(raws))
print(json.dumps(dict(samples=len(rows),raws=len(raws),seconds=time.monotonic()-start)),flush=True)

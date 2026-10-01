"""Passive reward-window capture plus sparse frame-matched observations."""
import json,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.telemetry_memory import read
from touhou_ai.live_inspect import verify_game
from touhou_ai.windows_probe import ReadOnlyProcess
from touhou_ai.th10_reader import Th10Reader
root=pathlib.Path(__file__).resolve().parents[1];art=root/'artifacts'
record=json.loads((root/'.runtime/live-learning.json').read_text());folder=pathlib.Path(record['Output'])
out=art/'boss-reward-tanh-20260930';out.mkdir(exist_ok=True)
previous=None;ui_previous=None;rows={};timeline=[];raws=[];proc=None;pid=None
start=time.monotonic();next_ui=0;next_raw=0
pending_raw=None
if (out/'capture.json').exists():
 old=json.loads((out/'capture.json').read_text())
 if old['run']['RunId']==record['RunId']:
  rows={(r['episode'],r['time']):r for r in old['rows']};timeline=old['timeline'];raws=old['observations']
try:
 while time.monotonic()-start<240:
  now=time.monotonic()
  if now>=next_ui:
   next_ui=now+.5;item=read(art/'ui-stats',previous=ui_previous)
   if item:
    ui_previous,payload=item;s=json.loads(payload)
    if s['run']==record['RunId']:
     for r in s['rows']:rows[(r['episode'],r['time'])]=r
  item=read(art,previous=previous)
  if item:
   previous,payload=item;t=json.loads(payload);l=t.get('learning') or {};g=t.get('game') or {}
   if l.get('phase')=='playing' and t.get('episode_id','').startswith(record['RunId']):
    boss=g.get('stage')==1 and t.get('game_frame',0)>=4000 and any(e.get('is_boss') for e in (t.get('entities') or {}).get('enemies',[]))
    timeline.append(dict(time=t['timestamp'],episode=t['episode_id'],boss=boss,stage=g.get('stage'),frame=t['game_frame'],steps=l['steps'],updates=l['updates']))
    if pending_raw and pending_raw['raw']['stage_frame']==t['game_frame'] and pending_raw['raw']['stage']==g['stage'] and pending_raw['episode']==t['episode_id']:
     raws.append(dict(pending_raw,time=t['timestamp'],rewards=t['reward']['components'],bomb_clock=t['policy'].get('bomb_clock')))
     pending_raw=None
    if boss and now>=next_raw:
     next_raw=now+.5
     try:
      status=json.loads((folder/'status.json').read_text());newpid=status['game_pid']
      if pid!=newpid:
       if proc:proc.close()
       config=verify_game(newpid);proc=ReadOnlyProcess(newpid,config['executable']);reader=Th10Reader(proc);pid=newpid
      before=reader.globals();raw=reader.snapshot();after=reader.globals()
      if (before['stage']==after['stage']==raw['stage']==g['stage'] and before['stage_frame']==after['stage_frame']==raw['stage_frame'] and raw['player'] and status['updates']==l['updates']):
       checkpoint=str(folder/status['episodes'][-1]['checkpoint']) if status['episodes'] else record['ResumeCheckpoint']
       pending_raw=dict(episode=t['episode_id'],updates=l['updates'],checkpoint=checkpoint,raw=raw)
       if raw['stage_frame']==t['game_frame']:
        raws.append(dict(pending_raw,time=t['timestamp'],rewards=t['reward']['components'],bomb_clock=t['policy'].get('bomb_clock')));pending_raw=None
       next_raw=now+3
     except (ValueError,OSError):pass
  if len(raws)>=20:break
  time.sleep(.01)
finally:
 if proc:proc.close()
 (out/'capture.json').write_text(json.dumps(dict(run=record,rows=list(rows.values()),timeline=timeline,observations=raws)))
print(json.dumps(dict(rows=len(rows),telemetry=len(timeline),observations=len(raws),seconds=time.monotonic()-start)),flush=True)

"""Bounded passive capture of exact UI rows and sampled scene context."""
import json,pathlib,sys,time,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.telemetry_memory import read
root=pathlib.Path(__file__).resolve().parents[1]
out=root/'artifacts/hit-windows-20261001';out.mkdir(exist_ok=True)
if (out/'status.json').exists():raise RuntimeError('Capture already exists; inspect before restarting')
start=time.time();ui_prev=scene_prev=None;last_ui=last_flush=0.;last_hit=None
seen=set();hits=0;rows_count=contexts_count=0;pending=[];contexts=[]
def flush(state):
    global last_flush
    for name,items in [('rows.jsonl',pending),('contexts.jsonl',contexts)]:
        if items:
            with (out/name).open('a',encoding='utf-8') as f:
                for item in items:f.write(json.dumps(item)+'\n')
            items.clear()
    status=dict(state=state,pid=os.getpid(),started=start,updated=time.time(),seconds=time.time()-start,hits=hits,target=30,rows=rows_count,contexts=contexts_count,
        context_sampling_seconds=.1,ui_poll_seconds=.5,maximum_seconds=7200,game_control=False)
    tmp=out/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(out/'status.json');last_flush=time.time()
flush('running')
try:
    while time.time()-start<7200:
        now=time.time()
        if now-last_ui>=.5:
            last_ui=now;item=read(root/'artifacts/ui-stats',previous=ui_prev)
            if item:
                ui_prev,payload=item;s=json.loads(payload)
                for r in s['rows']:
                    key=(s['run'],str(r['episode']),r['time'])
                    if key in seen or r['time']<start:continue
                    seen.add(key);pending.append(dict(r,run=s['run']));rows_count+=1
                    if r['components'].get('hit',0)<0:hits+=1;last_hit=r['time']
                seen={k for k in seen if k[2]>now-60}
        item=read(root/'artifacts',previous=scene_prev)
        if item:
            scene_prev,payload=item;t=json.loads(payload)
            if (t.get('learning') or {}).get('phase')=='playing':
                c={k:t.get(k) for k in ['timestamp','episode_id','game_frame','game','player','bomb','spell','applied_input','policy','learning','metrics']}
                es=t.get('entities') or {};c['bullet_count']=len(es.get('bullets') or []);c['laser_count']=len(es.get('lasers') or [])
                c['bosses']=[e for e in es.get('enemies') or [] if e.get('is_boss')]
                contexts.append(c);contexts_count+=1
        if now-last_flush>=30:flush('running')
        if hits>=30 and last_hit is not None and now-last_hit>=6:break
        time.sleep(.1)
    flush('complete' if hits>=30 else 'time_limit')
except BaseException:
    flush('failed');raise

"""Passive compact shared-memory capture; no game input or model changes."""
import sys,pathlib,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.telemetry_memory import read
root=pathlib.Path('artifacts');dest=root/'hit-td-20261001';dest.mkdir(exist_ok=True)
rows={};previous=None;start=time.monotonic()
while time.monotonic()-start<180:
    item=read(root/'ui-stats',previous=previous)
    if item:
        previous,payload=item;s=json.loads(payload)
        for r in s['rows']:rows[(s['run'],r['episode'],r['time'])]=dict(r,run=s['run'])
    time.sleep(.5)
(dest/'capture.json').write_text(json.dumps(list(rows.values())),encoding='utf-8')
print('captured',len(rows),'hit rows',sum(r['components'].get('hit',0)<0 for r in rows.values()),flush=True)

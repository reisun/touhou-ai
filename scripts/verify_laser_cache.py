import json,pathlib,sys,time,statistics,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from unittest.mock import patch
import touhou_ai.dual_grid as dg
from touhou_ai.live_action_grid import LiveActionGridContract
from touhou_ai.telemetry_memory import read
out=pathlib.Path('artifacts/laser-cache-20261003');out.mkdir(exist_ok=True)
source=subprocess.check_output(['git','-c','safe.directory=D:/repos/touhou-ai','show','604f6b6:touhou_ai/dual_grid.py'],text=True)
ns=dict(dg.__dict__);exec(source[source.index('def paint_laser('):source.index('\n\ndef bin_entities')],ns);old=ns['paint_laser']
raws=json.loads(pathlib.Path('artifacts/fps-diagnosis-20260930/raws.json').read_text());env=LiveActionGridContract();timings=[]
for item in raws:
 raw=item['raw'];new=env.encode(raw)
 with patch.object(dg,'paint_laser',old):ref=env.encode(raw)
 for k in ref:np.testing.assert_array_equal(ref[k],new[k])
 if not any(l['collision']['active'] for l in raw['lasers']):continue
 tt={'old':[],'warm':[],'cold':[]}
 for _ in range(5):
  for mode in tt:
   if mode=='cold':dg.laser_patch.cache_clear()
   with patch.object(dg,'paint_laser',old if mode=='old' else dg.paint_laser):
    t=time.perf_counter();env.encode(raw);tt[mode].append((time.perf_counter()-t)*1000)
 timings.append({k:statistics.median(v) for k,v in tt.items()})
# Passive geometry snapshots: no game memory access, inference, or game control.
seq=None;frames=[];start=time.monotonic()
while time.monotonic()-start<60:
 item=read(previous=seq)
 if item:
  seq,payload=item;d=json.loads(payload)
  if d.get('player') and (d.get('learning') or {}).get('phase')=='playing':
   lasers=[x['collision'] for x in d['entities']['lasers'] or [] if x.get('collision')]
   frames.append({'pos':d['player']['position'],'lasers':lasers,'frame':d['game_frame']})
 time.sleep(.04)
results={};dg.laser_patch.cache_clear()
for mode,fn in [('old',old),('cached',dg.paint_laser)]:
 t=time.perf_counter()
 for f in frames:
  local=np.zeros((6,96,96),np.float32);whole=np.zeros((12,56,48),np.float32)
  origin=np.array(f['pos'])-[192,0]-96
  for c in f['lasers']:
   fn(local,4,5,c,origin,2);fn(whole,8,9,c,np.array([-192,0]),8)
 results[mode]=(time.perf_counter()-t)*1000
report={'matched_full_observations':len(raws),'repeated_scene_encode_median_ms':{k:statistics.median(x[k] for x in timings) for k in timings[0]},'passive_frames':len(frames),'laser_frames':sum(any(c['active'] for c in f['lasers']) for f in frames),'sequential_laser_paint_ms':results,'cache':dg.laser_patch.cache_info()._asdict()}
(out/'report.json').write_text(json.dumps(report,indent=2));(out/'geometry.json').write_text(json.dumps(frames));print(json.dumps(report),flush=True)

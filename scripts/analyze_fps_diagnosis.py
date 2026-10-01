"""Analyze passive frame progression and replay CPU stages on captured observations."""
import json,pathlib,time,sys,statistics
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.live_action_grid import LiveActionGridContract,live_risk_grid
from touhou_ai.dual_grid import DualGridContract
from touhou_ai.telemetry import packet
from touhou_ai.checkpoint_rng import load_preserving_rng
from touhou_ai.separate_clip_ppo import SeparateClipPPO
root=pathlib.Path(__file__).resolve().parents[1];out=root/'artifacts/fps-diagnosis-20260930'
rows=json.loads((out/'telemetry.json').read_text());pairs=[]
for a,b in zip(rows,rows[1:]):
 dt=b['time']-a['time'];ds=b['steps']-a['steps']
 if a['episode']==b['episode'] and a['stage']==b['stage'] and a['phase']==b['phase'] and 0<dt<1 and 0<ds<=10 and b['frame']-a['frame']==ds*2:
  pairs.append(dict(phase=b['phase'],dt=dt,ds=ds,bullets=b['bullets'],ms=dt/ds*1000,sample_ms=b['sample_ms']))
summary={}
for phase in ('road','boss','other'):
 a=[x for x in pairs if x['phase']==phase]
 if a:summary[phase]=dict(intervals=len(a),seconds=sum(x['dt'] for x in a),effective_fps=2*sum(x['ds'] for x in a)/sum(x['dt'] for x in a),mean_bullets=statistics.mean(x['bullets'] for x in a),median_action_ms=statistics.median(x['ms'] for x in a),median_observe_ms=statistics.median(x['sample_ms'] for x in a))
print('LIVE',json.dumps(summary),flush=True)
torch.set_num_threads(1)
r=json.loads((out/'run.json').read_text());folder=pathlib.Path(r['Output']);s=json.loads((folder/'status.json').read_text());model=load_preserving_rng(SeparateClipPPO,folder/s['episodes'][-1]['checkpoint'],device='cpu');model.policy.set_training_mode(False)
env=LiveActionGridContract();base=DualGridContract();bench=[]
def measure(fn):
 fn();times=[]
 for _ in range(7):
  t=time.perf_counter();fn();times.append((time.perf_counter()-t)*1000)
 return statistics.median(times)
with torch.no_grad():
 for item in json.loads((out/'raws.json').read_text()):
  raw=item['raw'];obs=env.encode(raw)
  def infer():return model.policy.forward_with_distribution(model.policy.obs_to_tensor(obs)[0])
  def telemetry():
   t=packet(raw,'benchmark','live');t['ai_observation']={'bullets':t['entities']['bullets'],'items':t['entities']['items'],'bullet_grid':[]};return t
  t=telemetry()
  bench.append(dict(phase=item['phase'],frame=raw['stage_frame'],bullets=len(raw['bullets']),items=len(raw['items']),encode_ms=measure(lambda:env.encode(raw)),base_encode_ms=measure(lambda:base.encode(raw)),risk_ms=measure(lambda:live_risk_grid(raw)),inference_ms=measure(infer),packet_ms=measure(telemetry),json_ms=measure(lambda:json.dumps(t,allow_nan=False,separators=(',',':'))),json_bytes=len(json.dumps(t,separators=(',',':')))))
for phase in ('road','boss','other'):
 a=[x for x in bench if x['phase']==phase]
 if a:print('BENCH',phase,len(a),json.dumps({k:statistics.median(x[k] for x in a) for k in ['bullets','encode_ms','base_encode_ms','risk_ms','inference_ms','packet_ms','json_ms','json_bytes']}),flush=True)
(out/'analysis.json').write_text(json.dumps(dict(live=summary,bench=bench),indent=2))

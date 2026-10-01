"""Passive memory/telemetry capture. No lease, hook, input, or publication."""
import sys,pathlib,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.telemetry_memory import read
from touhou_ai.live_inspect import verify_game
from touhou_ai.windows_probe import ReadOnlyProcess
from touhou_ai.th10_reader import Th10Reader
out=pathlib.Path('artifacts/live-action-diagnosis-20260929');out.mkdir(exist_ok=True)
run=json.loads(pathlib.Path('.runtime/live-learning.json').read_text())
status=json.loads((pathlib.Path(run['Output'])/'status.json').read_text())
(out/'current-status.json').write_text(json.dumps(status))
(out/'run.json').write_text(json.dumps(run))
config=verify_game(status['game_pid']);process=ReadOnlyProcess(status['game_pid'],config['executable']);reader=Th10Reader(process)
samples=[];raws=[];previous=None;next_raw=0;start=time.monotonic();races=0
try:
    while time.monotonic()-start<100:
        item=read(previous=previous)
        if item:
            previous,payload=item;d=json.loads(payload)
            if d.get('episode_id','').startswith(run['RunId']) and d.get('learning',{}).get('phase')=='playing':
                samples.append({k:d.get(k) for k in ('timestamp','episode_id','game_frame','game','player','policy','reward','applied_input','learning','metrics','entities','ai_observation')})
        if time.monotonic()>=next_raw:
            next_raw=time.monotonic()+.5
            try:
                before=reader.globals()['stage_frame'];raw=reader.snapshot();after=reader.globals()['stage_frame']
                if before==after==raw['stage_frame'] and raw['player'] and raw['lives_raw']>=0:
                    raws.append(raw)
                else:races+=1
            except (ValueError,OSError):races+=1
        time.sleep(.05)
finally:process.close()
(out/'telemetry.json').write_text(json.dumps(samples))
(out/'raws.json').write_text(json.dumps(raws))
(out/'capture.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,telemetry_samples=len(samples),raw_samples=len(raws),discarded_races=races,input_control=False)))
print('CAPTURE',len(samples),len(raws),races,flush=True)

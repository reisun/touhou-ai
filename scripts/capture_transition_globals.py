"""Read-only transition diagnostics; no input or hooks."""
import pathlib,json,time,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.live_inspect import verify_game
from touhou_ai.windows_probe import ReadOnlyProcess
from touhou_ai.th10_reader import Th10Reader
root=pathlib.Path(__file__).resolve().parents[1]
out=root/'artifacts/stage-transition-fix-20260930';out.mkdir(exist_ok=True)
record=json.loads((root/'.runtime/live-learning.json').read_text())
folder=pathlib.Path(record['Output'])
process=None;pid=None;last=None;deadline=time.monotonic()+600
try:
 with (out/'globals.jsonl').open('a') as log:
  while time.monotonic()<deadline:
   s=json.loads((folder/'status.json').read_text())
   if s.get('game_pid')!=pid:
    if process:process.close()
    pid=s['game_pid'];c=verify_game(pid);process=ReadOnlyProcess(pid,c['executable']);reader=Th10Reader(process)
   try:
    g=reader.globals();key=tuple(g.get(k) for k in ['character','shot','difficulty','stage','mode_flags'])
    if key!=last:
     row=dict(time=time.time(),pid=pid,globals=g);log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True);last=key
   except OSError:pass
   time.sleep(.02)
finally:
 if process:process.close()

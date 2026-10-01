"""Guarded checkpoint archive for an explicitly selected live campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

parser=argparse.ArgumentParser()
parser.add_argument('--expected-run', required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
record=json.loads((root/'.runtime/live-learning.json').read_text(encoding='utf-8-sig'))
assert record['RunId']==args.expected_run
folder=(root/'artifacts'/args.expected_run).resolve()
assert folder.parent== (root/'artifacts').resolve() and folder==Path(record['Output']).resolve()
out=args.output.resolve()
assert out.parent==(root/'artifacts').resolve()
out.mkdir(exist_ok=False)
(out/'previous-run.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
deadline=time.monotonic()+900
requested=False
while time.monotonic()<deadline:
    status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if status['status']=='failed':raise RuntimeError(status.get('error','learner failed'))
    if status['status']=='stopped':break
    if status.get('phase')=='optimizing_at_game_over' and not requested:
        assert json.loads((root/'.runtime/live-learning.json').read_text(encoding='utf-8-sig'))==record
        (folder/'STOP').touch();requested=True
        print('Requested stop at completed-game boundary',flush=True)
    time.sleep(.25)
else:raise TimeoutError('No boundary within 900s')
last=status['episodes'][-1]
assert last['reload_verified'] and last['updated_at_game_over'] and not last['truncated']
source=folder/last['checkpoint'];digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==last['checkpoint_sha256']
target=out/'preserved-model.zip';shutil.copy2(source,target)
assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
(out/'previous-final-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
manifest=dict(source=str(source),preserved=str(target),sha256=digest,updates=status['updates'])
(out/'archive.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest),flush=True)

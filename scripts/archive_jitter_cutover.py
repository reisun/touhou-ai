"""One-time, guarded game-over cutover for the approved fresh jitter campaign."""
import hashlib
import json
from pathlib import Path
import shutil
import time

root = Path(__file__).resolve().parents[1]
out = root/'artifacts/jitter-adoption-20261001'
record = json.loads((out/'previous-run.json').read_text(encoding='utf-8-sig'))
assert record == json.loads((root/'.runtime/live-learning.json').read_text(encoding='utf-8-sig'))
assert record['RunId'] == 'live-learning-20261001-190603-3fbec1'
folder = Path(record['Output'])
deadline = time.monotonic()+900
requested = False
while time.monotonic() < deadline:
    status = json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if status['status'] == 'failed':
        raise RuntimeError(status.get('error', 'learner failed'))
    if status['status'] == 'stopped':
        break
    if status.get('phase') == 'optimizing_at_game_over' and not requested:
        (folder/'STOP').touch()
        requested = True
        print('STOP requested at game-over boundary', flush=True)
    time.sleep(.25)
else:
    raise TimeoutError('No game-over boundary within 900 seconds')
last = status['episodes'][-1]
assert last['reload_verified'] and last['updated_at_game_over'] and not last['truncated']
source = folder/last['checkpoint']
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == last['checkpoint_sha256']
target = out/'preserved-model.zip'
if target.exists():
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
else:
    shutil.copy2(source, target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
(out/'previous-final-status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
manifest = dict(source=str(source), preserved=str(target), sha256=digest,
                updates=status['updates'], all_original_checkpoints_retained=True)
(out/'archive.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest), flush=True)

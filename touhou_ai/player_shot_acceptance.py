"""Bounded native shot/grid check, with no model or optimizer updates."""
import json
from pathlib import Path
import numpy as np
from touhou_ai.live_runtime import LiveRuntime
from touhou_ai.live_reset import continue_episode
from touhou_ai.live_acceptance import pause
from touhou_ai.dual_grid import DualGridContract, CONTRACT, GLOBAL_CHANNELS


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    pid=json.loads(Path('.runtime/game.json').read_text(encoding='utf-8-sig'))['Id']
    runtime=LiveRuntime(pid, Path(__file__).with_name('player_shot_trace.js'))
    report={'status':'running','contract':CONTRACT,'training_updates':0}
    try:
        continue_episode(runtime)
        env=DualGridContract(); maxima={}; sizes=set(); moved=0; previous={}
        with (output/'trace.jsonl').open('w',encoding='utf-8') as stream:
            for i in range(480):
                phase='neutral' if i<60 else 'shoot' if i<260 else 'focus' if i<380 else 'off'
                mask=0 if phase in ('neutral','off') else 1 if phase=='shoot' else 5
                raw=runtime.step_gameplay(mask,2)
                obs=env.encode(raw)
                shots=raw['player_shots']; grid=obs['global_grid'][GLOBAL_CHANNELS.index('player_shot_coverage')]
                assert np.isfinite(grid).all() and grid.min()>=0 and grid.max()<=1
                if phase=='neutral': assert not shots and not grid.any()
                maxima[phase]=max(maxima.get(phase,0),len(shots))
                for shot in shots:
                    sizes.add(tuple(shot['hitbox_raw']))
                    old=previous.get(shot['slot'])
                    if old and shot['position'][1]<old[1]: moved+=1
                previous={s['slot']:s['position'] for s in shots}
                stream.write(json.dumps({'phase':phase,'raw':raw,'shot_grid_sum':float(grid.sum())})+'\n')
                if i%120==0: print(json.dumps({'sample':i,'phase':phase,'shots':len(shots)}),flush=True)
                if raw['lives_raw']<0: raise AssertionError('unexpected early game over')
            contacts=runtime.api.shotcontacts()
            assert maxima['shoot']>0 and maxima['focus']>0 and moved>0
            assert not shots and not grid.any(), 'shots did not clear after release'
            assert contacts, 'no native shot contact evidence'
            report.update(status='passed',samples=480,max_shots=maxima,hitbox_sizes=sorted(sizes),
                          upward_motion_samples=moved,final_shots=len(shots),native_contacts=len(contacts))
            (output/'contacts.json').write_text(json.dumps(contacts,indent=2),encoding='utf-8')
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:
        try: pause(runtime)
        finally:
            runtime.close()
            (output/'status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)

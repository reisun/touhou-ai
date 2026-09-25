"""Explicit, bounded real-game probes. No stat writes or training updates."""
import argparse
import json
from pathlib import Path
import time

from touhou_ai.live_runtime import LiveRuntime
from touhou_ai.live_learning import atomic_json, publish_telemetry
from touhou_ai.telemetry import packet


def pause(runtime):
    state = runtime.snapshot(full=False)
    if state["lives_raw"] < 0:
        return state
    if state["pause_words"][1] == 0:
        runtime.step(8, 1, full=False)
    for _ in range(120):
        state = runtime.step(0, 1, full=False)
        if state["pause_words"][1] == 2:
            return state
    raise RuntimeError("could not return to pause")


def probe(output):
    output.mkdir(parents=True, exist_ok=False)
    record = json.loads(Path('.runtime/game.json').read_text(encoding='utf-8-sig'))
    runtime = LiveRuntime(record['Id'])
    report = {"status": "running", "pid": record['Id'], "training_steps": 0}
    timings, rows = [], []
    def sample(mask=0, frames=2):
        started = time.perf_counter()
        state = runtime.step(mask, frames)
        timings.append(time.perf_counter()-started)
        rows.append(state)
        publish_telemetry(packet(state, output.name, 'live'))
        if state['lives_raw'] < 0:
            raise RuntimeError('probe reached game over')
        return state
    try:
        initial = runtime.snapshot()
        atomic_json(output/'initial.json', initial)
        words = initial['pause_words']
        if (words is None or words[0:2] != [2, 2] or words[9] != 0
                or initial['bomb']['state'] != 0 or initial['power_raw'] < 20):
            raise ValueError('requires paused Resume selection and one available bomb')
        runtime.step(1, 1, full=False)
        for _ in range(120):
            state = sample(0, 1)
            if state['pause_words'][1] == 0 and state['stage_frame'] > initial['stage_frame']:
                break
        else:
            raise TimeoutError('pause did not resume')
        for _ in range(30):
            state = sample(1)
        before = state
        after = sample(2)
        if before['bomb']['state'] != 0 or after['bomb']['state'] != 1:
            raise AssertionError('bomb activation was not observed')
        if before['power_raw']-after['power_raw'] != 20:
            raise AssertionError('expected one power unit consumed by bomb')
        report['activation'] = {'frame': after['stage_frame'], 'before_power': before['power_raw'],
                                'after_power': after['power_raw'], 'state': after['bomb']}
        for _ in range(300):
            state = sample(0)
            if state['bomb']['state'] == 0:
                break
        else:
            raise TimeoutError('bomb did not finish within 600 frames')
        report['bomb_end_frame'] = state['stage_frame']
        if state['power_raw'] < 20:
            before = state
            state = sample(2)
            if state['bomb']['state'] != 0 or state['power_raw'] < before['power_raw']:
                raise AssertionError('insufficient-power request was not a no-op')
            report['insufficient_power_noop'] = True
        hp = [e['hp'] for s in rows for e in s['enemies'] or [] if e['is_boss']]
        report.update(status='passed', boss_hp_min=min(hp) if hp else None,
                      boss_hp_max=max(hp) if hp else None,
                      acceleration_samples=sum(b['acceleration'] is not None for s in rows for b in s['bullets'] or []),
                      sample_hz=len(timings)/sum(timings))
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        try:
            report['final_pause'] = pause(runtime)
            publish_telemetry(packet(report['final_pause'], output.name, 'live'))
        finally:
            runtime.close()
            with (output/'samples.jsonl').open('x', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row)+'\n')
            atomic_json(output/'status.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    probe(parser.parse_args().output)

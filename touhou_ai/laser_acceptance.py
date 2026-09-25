"""One-life, bounded line-laser probe, with actual collision-call evidence."""
import argparse
import json
import math
from pathlib import Path
import time

from touhou_ai.live_features import laser_collision


def assess(row):
    rectangle = laser_collision('line', row['state'], row['position'], *row['raw'])
    error = max(abs(a-b) for a, b in zip(
        [*rectangle['origin'], rectangle['angle'], rectangle['width'], rectangle['length']],
        [*row['origin'], row['angle'], row['width'], row['length']]))
    dx, dy = [a-b for a, b in zip(row['player'], row['origin'])]
    c, s = math.cos(row['angle']), math.sin(row['angle'])
    x, y = dx*c+dy*s, -dx*s+dy*c
    hx, hy = row['halfbox']
    margin = max(-x-hx, x-hx-row['length'], abs(y)-hy-row['width']/2)
    eligible = row['player_status'] not in (2, 3, 4) and row['invincibility'] <= 0 and not row['dialogue']
    expected = margin <= 0 and eligible
    return {'parameter_error': error, 'margin': margin, 'eligible': eligible,
            'expected_hit': expected, 'actual_hit': row['result'] == 1,
            'mismatch': abs(margin) > 0.0001 and expected != (row['result'] == 1)}


def approach(state):
    x, y = state['player']['position']
    candidates = []
    for beam in state['lasers'] or []:
        rect = beam.get('collision')
        if beam['kind'] != 'line' or not rect or not rect['active']:
            continue
        ox, oy = rect['origin']
        c, s = math.cos(rect['angle']), math.sin(rect['angle'])
        along = max(0, min(rect['length'], (x-ox)*c+(y-oy)*s))
        target = (ox+along*c, oy+along*s)
        if -180 <= target[0] <= 180 and 32 <= target[1] <= 432:
            candidates.append(((x-target[0])**2+(y-target[1])**2, target))
    if not candidates:
        return 0
    _, (tx, ty) = min(candidates)
    return 4 | (0x40 if tx < x-1 else 0x80 if tx > x+1 else 0) | (0x10 if ty < y-1 else 0x20 if ty > y+1 else 0)


def probe(output):
    from touhou_ai.live_runtime import LiveRuntime
    from touhou_ai.live_acceptance import pause
    from touhou_ai.live_learning import atomic_json, publish_telemetry
    from touhou_ai.telemetry import packet
    output.mkdir(parents=True, exist_ok=False)
    record = json.loads(Path('.runtime/game.json').read_text(encoding='utf-8-sig'))
    runtime = trace = None
    report = {'status': 'running', 'training_updates': 0, 'pid': record['Id']}
    calls, states = [], []
    started = time.monotonic()
    try:
        runtime = LiveRuntime(record['Id'])
        initial = runtime.snapshot()
        atomic_json(output/'initial.json', initial)
        if (initial['pause_words'][:2] != [2, 2] or initial['pause_words'][9] != 0
                or initial['replay_mode'] != 0 or initial['mode_flags'] != 0
                or initial['lives_raw'] < 1 or not initial['lasers']):
            raise ValueError('requires paused gameplay with lasers and reserve lives')
        trace = runtime.session.create_script(Path(__file__).with_name('laser_trace.js').read_text())
        trace.load()
        def collect(mask):
            state = runtime.step(mask, 1)
            batch = trace.exports_sync.drain()
            if batch['overflow'] or batch['fault']:
                raise RuntimeError(str(batch))
            calls.extend(batch['rows'])
            states.append(state)
            if len(states) % 10 == 0:
                publish_telemetry(packet(state, output.name, 'live'))
            return state
        state = collect(1)
        stop_reason = 'frame_budget'
        for _ in range(900):
            if time.monotonic()-started > 90 or (output/'STOP').exists():
                stop_reason = 'wall_time_or_stop'
                break
            if state['lives_raw'] < initial['lives_raw']:
                stop_reason = 'one_life_lost'
                break
            if state['stage'] != initial['stage'] or state['replay_mode'] != 0:
                stop_reason = 'stage_or_mode_transition'
                break
            state = collect(approach(state) if state['pause_words'][1] == 0 else 0)
        results = [assess(row) for row in calls]
        eligible = [r for r in results if r['eligible']]
        report.update(stop_reason=stop_reason, calls=len(calls),
            actual_laser_hits=sum(r['actual_hit'] for r in results),
            mismatches=sum(r['mismatch'] for r in results),
            max_parameter_error=max((r['parameter_error'] for r in results), default=None),
            nearest_outside=min((r['margin'] for r in eligible if r['margin'] > 0), default=None),
            nearest_inside=max((r['margin'] for r in eligible if r['actual_hit']), default=None),
            initial_lives=initial['lives_raw'], final_lives=state['lives_raw'],
            elapsed=time.monotonic()-started)
        report['status'] = ('matched_samples' if calls and report['actual_laser_hits']
                            and not report['mismatches'] and report['max_parameter_error'] < .001
                            else 'incomplete_or_mismatch')
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        try:
            if runtime is not None:
                report['final_pause'] = pause(runtime)
                publish_telemetry(packet(runtime.snapshot(), output.name, 'live'))
        finally:
            try:
                if trace is not None:
                    trace.exports_sync.dispose()
                    trace.unload()
            finally:
                try:
                    if runtime is not None:
                        runtime.close()
                finally:
                    for filename, rows in [('calls.jsonl', calls), ('states.jsonl', states)]:
                        with (output/filename).open('x', encoding='utf-8') as stream:
                            for row in rows:
                                stream.write(json.dumps(row, allow_nan=False)+'\n')
                    atomic_json(output/'status.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'final_pause'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    probe(parser.parse_args().output)

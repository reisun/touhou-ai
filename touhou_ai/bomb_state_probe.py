"""Bounded bomb-state diagnostic; no model updates or game-stat writes."""
import argparse
import json
from pathlib import Path

from touhou_ai.live_runtime import LiveRuntime
from touhou_ai.live_reset import continue_episode, resume_paused_episode
from touhou_ai.live_acceptance import pause


def run(output, pid):
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'training_updates': 0, 'pid': pid}
    runtime = LiveRuntime(pid)
    keys = ('stage_frame', 'gate_tick', 'bomb', 'power_raw', 'lives_raw',
            'player', 'input_state_raw', 'combat_reward_events', 'pause_words')
    with (output / 'trace.jsonl').open('x', encoding='utf-8') as stream:
        def record(state, phase, mask=0):
            stream.write(json.dumps({'phase': phase, 'mask': mask,
                                     **{k: state.get(k) for k in keys}}) + '\n')
            stream.flush()
            return state
        def step(mask, frames, phase, full=False):
            state = record(runtime.step_gameplay(mask, frames, full), phase, mask)
            if state.get('transition') != 'gameplay':
                raise RuntimeError('diagnostic reached terminal/transition')
            return state
        try:
            state = record(runtime.snapshot(), 'initial')
            if state['lives_raw'] == -1:
                state = continue_episode(runtime)
            elif state['pause_words'][1] == 2:
                state = resume_paused_episode(runtime)
            state = record(state, 'ready')
            # Collect actual dropped items with ordinary movement and shot input.
            for index in range(450):
                if state['power_raw'] >= 20:
                    break
                state = runtime.snapshot()
                x, y = state['player']['position']
                items = [i for i in state['items'] or [] if 0 < i['position'][1] < 440]
                target = min(items, key=lambda i: abs(i['position'][0]-x)+abs(i['position'][1]-y)) if items else None
                tx, ty = target['position'] if target else (192, 360)
                mask = 1 | (0x80 if tx-x > 10 else 0x40 if x-tx > 10 else 0)
                mask |= 0x20 if ty-y > 10 else 0x10 if y-ty > 10 else 0
                state = step(mask, 8, 'acquire_power')
            if state['power_raw'] < 20:
                raise RuntimeError('no bomb power within acquisition budget')
            # Allow previously fired shots to leave before the measured bomb.
            for _ in range(90):
                state = step(0, 1, 'shot_washout')
            report['before'] = {k: state.get(k) for k in keys}
            state = step(2, 1, 'bomb_request')
            if state['bomb']['state'] != 1:
                raise AssertionError('bomb request did not activate')
            report['first_active_frame'] = state['stage_frame']
            report['activation_power_drop'] = report['before']['power_raw'] - state['power_raw']
            for _ in range(600):
                state = step(0, 1, 'bomb_window')
                if state['bomb']['state'] == 1:
                    report.setdefault('first_active_frame', state['stage_frame'])
                elif 'first_active_frame' in report:
                    report['first_inactive_frame'] = state['stage_frame']
                    break
            if 'first_inactive_frame' not in report:
                raise AssertionError('complete bomb cycle not observed')
            for _ in range(90):
                state = step(0, 1, 'post_bomb')
            if state['power_raw'] < 20:
                step(2, 1, 'insufficient_request')
                for _ in range(10):
                    state = step(0, 1, 'insufficient_followup')
            report['status'] = 'passed'
        except BaseException as error:
            report.update(status='failed', error=str(error))
            raise
        finally:
            try:
                report['final'] = record(pause(runtime), 'final_pause')
            finally:
                runtime.close()
                (output / 'status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    run(args.output, args.pid)

"""Passive 30 Hz telemetry: read-only process access, no Frida or input hook."""
import argparse
import json
from pathlib import Path
import time

from touhou_ai.live_inspect import verify_game
from touhou_ai.process_lease import ProcessLease
from touhou_ai.th10_reader import Th10Reader
from touhou_ai.windows_probe import ReadOnlyProcess
from touhou_ai.telemetry import packet


def observe(pid, seconds, output, model_monitor=False):
    monitor = None
    if model_monitor:
        from touhou_ai.model_monitor import ModelMonitor
        monitor = ModelMonitor()
    config = verify_game(pid)
    lease = ProcessLease(pid)
    process = ReadOnlyProcess(pid, config['executable'])
    reader = Th10Reader(process)
    output.mkdir(parents=True, exist_ok=False)
    latest = Path('artifacts/telemetry/latest.json')
    latest.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started+seconds
    count = dropped = 0
    logged_key, logged_bytes = None, 0
    status = {'status': 'running', 'pid': pid, 'read_only': True, 'input_control': False}
    try:
        with (output/'samples.jsonl').open('x', encoding='utf-8') as log:
            next_sample = time.monotonic()
            consecutive_errors = 0
            while time.monotonic() < deadline and not (output/'STOP').exists():
                sampled_at = time.perf_counter()
                try:
                    frame = reader.globals()['stage_frame']
                    state = reader.snapshot()
                    after = reader.globals()['stage_frame']
                    consecutive_errors = 0
                except (OSError, ValueError):
                    reader.previous_bullets = None
                    dropped += 1
                    consecutive_errors += 1
                    if consecutive_errors >= 30:
                        raise
                    time.sleep(1/30)
                    continue
                if frame != after or frame != state['stage_frame']:
                    reader.previous_bullets = None
                    dropped += 1
                else:
                    state.update(frame_locked=False, frame_stamp_unchanged=True,
                                 sample_ms=(time.perf_counter()-sampled_at)*1000)
                    data = packet(state, output.name, 'live')
                    if monitor is not None:
                        monitor.attach(state, data)
                    data['observer'] = {'mode': 'passive_read_only', 'samples': count+1,
                        'dropped_racing_frames': dropped, 'target_hz': 30,
                        'mean_hz': count/max(time.monotonic()-started, 0.001)}
                    payload = json.dumps(data, allow_nan=False, separators=(',', ':'))
                    temporary = latest.with_suffix('.tmp')
                    temporary.write_text(payload, encoding='utf-8')
                    try:
                        temporary.replace(latest)
                    except PermissionError:
                        dropped += 1
                    key = (state['stage'], state['stage_frame'], state['replay_mode'])
                    if key != logged_key and logged_bytes < 128*1024*1024:
                        line = json.dumps({'raw': state, 'telemetry': data}, separators=(',', ':'))+'\n'
                        log.write(line)
                        log.flush()
                        logged_bytes += len(line.encode('utf-8'))
                        logged_key = key
                    count += 1
                next_sample += 1/30
                time.sleep(max(0, next_sample-time.monotonic()))
                if next_sample < time.monotonic()-1/30:
                    next_sample = time.monotonic()
            status['status'] = 'stopped'
    except BaseException as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        process.close()
        lease.close()
        status.update(samples=count, dropped=dropped, elapsed=time.monotonic()-started,
                      log_limit_reached=logged_bytes >= 128*1024*1024)
        (output/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-monitor', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        raise ValueError('observation budget is 1..1800 seconds')
    observe(args.pid, args.seconds, args.output, args.model_monitor)

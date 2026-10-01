"""Exclusive, bounded frame stepping for the pinned TH10 executable."""
import argparse
import json
from pathlib import Path
import queue
import time
import hashlib

from touhou_ai.telemetry import packet

from touhou_ai.live_inspect import verify_game
from touhou_ai.th10_reader import Th10Reader
from touhou_ai.windows_probe import ReadOnlyProcess


def input_mask(action):
    if not isinstance(action, (list, tuple)) or len(action) != 4:
        raise ValueError("expected [direction, shoot, focus, bomb]")
    if any(type(value) is not int or not 0 <= value < limit for value, limit in zip(action, (9, 2, 2, 2))):
        raise ValueError("invalid action")
    direction, shoot, focus, bomb = action
    return (0, 0x10, 0x90, 0x80, 0xa0, 0x20, 0x60, 0x40, 0x50)[direction] | shoot | (focus << 2) | (bomb << 1)


class LiveRuntime:
    def __init__(self, pid, diagnostic_script=None):
        import frida
        config = verify_game(pid)
        data_path = Path(config['executable']).with_name('th10.dat')
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != '1fb1d0ffe34115f563f5feb43755c0feee2315b0ac2b32e2f9e84c81e9433bea':
            raise ValueError('unverified game data for progress rewards')
        self.process = ReadOnlyProcess(pid, config["executable"])
        self.session = self.script = None
        self.lease = None
        self.messages = queue.Queue()
        self.park = None
        try:
            from touhou_ai.process_lease import ProcessLease
            self.lease = ProcessLease(pid)
            self.session = frida.attach(pid)
            source = Path(__file__).with_name("th10_gate.js").read_text()
            spell_rows = json.loads(Path(__file__).with_name('spell_progress.json').read_text())
            source += '\nconst NORMAL_SPELL_PROGRESS = ' + json.dumps(spell_rows) + ';\n'
            source += '\n' + Path(__file__).with_name('progress_events.js').read_text(encoding='utf-8')
            if diagnostic_script is not None:
                source += '\n' + Path(diagnostic_script).read_text(encoding='utf-8')
            self.script = self.session.create_script(source)
            self.script.on("message", self._message)
            self.script.load()
            self.api = self.script.exports_sync
            self.reader = Th10Reader(self.process)
            self.api.arm()
            self.park = self._wait()
        except BaseException:
            self.close()
            raise

    def _message(self, message, data):
        if message["type"] == "error":
            self.messages.put(RuntimeError(message.get("description", "hook failure")))
        elif message["type"] == "send" and message["payload"].get("type") == "parked":
            self.messages.put(message["payload"])

    def _wait(self):
        try:
            message = self.messages.get(timeout=4)
        except queue.Empty:
            raise TimeoutError("game frame gate stalled") from None
        if isinstance(message, Exception):
            raise message
        return message

    def snapshot(self, full=True):
        status = self.api.status()
        if not status["enabled"] or not status["parked"] or status["tick"] != self.park["tick"]:
            raise RuntimeError("game not held at expected frame")
        reader_started = time.perf_counter()
        result = self.reader.snapshot(full)
        reader_ms = (time.perf_counter()-reader_started)*1000
        combat = self.api.combat()
        if combat['error']:
            raise RuntimeError('combat observation failed: ' + combat['error'])
        result['combat_reward_events'] = combat['events']
        progress = self.api.progress()
        if progress['error']:
            raise RuntimeError('progress observation failed: ' + progress['error'])
        result['progress_reward_events'] = progress['events']
        result['reward_events_verified'] = True
        after = self.api.status()
        if not after["parked"] or not after["enabled"] or after["tick"] != status["tick"]:
            raise RuntimeError("snapshot lease lost")
        return result | {"gate_tick": status["tick"], "frame_locked": True,
                         "input_calls": after["input_calls"], "snapshot_full": full,
                         "reader_snapshot_ms": reader_ms}

    def owns_full_snapshot(self, state):
        if (state.get('snapshot_full') is not True or state.get('frame_locked') is not True
                or state.get('gate_tick') != self.park['tick']):
            return False
        status = self.api.status()
        return (status['enabled'] and status['parked']
                and status['tick'] == state['gate_tick'])

    def step(self, mask=0, frames=2, full=True, gameplay_guard=False):
        if (type(mask) is not int or mask < 0 or mask & ~0xff
                or type(frames) is not int or not 1 <= frames <= 120):
            raise ValueError("invalid frame action")
        start = self.park["tick"]
        advance_started = time.perf_counter()
        self.api.step(start, frames, mask, gameplay_guard)
        self.park = self._wait()
        if self.park["tick"] != start+frames:
            raise RuntimeError("frame gate skipped updates")
        advance_ms = (time.perf_counter()-advance_started)*1000
        snapshot_started = time.perf_counter()
        state = self.snapshot(full)
        state["runtime_timing_ms"] = {"advance_wait_ms": advance_ms,
            "snapshot_ms": (time.perf_counter()-snapshot_started)*1000,
            "reader_snapshot_ms": state['reader_snapshot_ms']}
        return state

    def step_gameplay(self, mask=0, frames=2, full=True):
        guard_started = time.perf_counter()
        before = self.snapshot(full=False)
        guard_ms = (time.perf_counter()-guard_started)*1000
        if (before["replay_mode"] != 0 or before["mode_flags"] not in (0, 4) or before["player"] is None
                or before.get('stage_init_pending', False) is not False):
            raise RuntimeError("not normal live gameplay")
        state = self.step(mask, frames, full, gameplay_guard=True)
        state.setdefault("runtime_timing_ms", {})["guard_snapshot_ms"] = guard_ms
        if (state["stage"] != before["stage"] or state["player"] is None
                or state["lives_raw"] < 0):
            state["transition"] = "terminal_or_stage_change"
        elif state["stage_frame"]-before["stage_frame"] != frames:
            keys = ('stage', 'stage_frame', 'mode_flags', 'screen_state_raw',
                    'pause_words', 'stage_init_pending', 'stage_manager_raw')
            detail = {name: {key: value.get(key) for key in keys}
                      for name, value in [('before', before), ('after', state)]}
            raise RuntimeError(f"gameplay frame mismatch (pause/loading/stall): {detail}")
        else:
            state["transition"] = "gameplay"
        return state

    def close(self):
        if self.script is not None:
            try:
                self.script.exports_sync.dispose()
            except Exception:
                pass
        try:
            if self.session is not None:
                self.session.detach()
        finally:
            self.process.close()
            if self.lease is not None:
                self.lease.close()
            self.script = self.session = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tap", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--movement-test", action="store_true")
    parser.add_argument("--watchdog-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1800:
        raise ValueError("diagnostic step budget must be 1..1800")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        runtime = LiveRuntime(args.pid)
        status = {"status": "running", "gameplay_training_steps": 0, "pid": args.pid}
        try:
            if args.start:
                from touhou_ai.live_reset import start_episode
                status["reset"] = start_episode(runtime)
                print(json.dumps({"reset": status["reset"]}))
            if args.movement_test:
                before = runtime.snapshot(full=False)
                if before["player"] is None or before["replay_mode"] != 0 or before["mode_flags"] not in (0, 4):
                    raise ValueError("movement test requires normal live gameplay")
                for mask in (0x80, 0x40, 0x04, 0x01, 0):
                    after = runtime.step_gameplay(mask, 10, full=False)
                    dx = after["player"]["position"][0]-before["player"]["position"][0]
                    if (mask == 0x80 and dx <= 0) or (mask == 0x40 and dx >= 0):
                        raise AssertionError("direction input did not move the player")
                    if after["input_state_raw"][0] != mask:
                        raise AssertionError("game input state does not match command")
                    if mask == 4 and after["player"]["focus_raw"] != 1:
                        raise AssertionError("focus input was not applied")
                    stream.write(json.dumps(after)+"\n")
                    print(json.dumps({"mask": mask, "state": after}))
                    before = after
            if args.tap:
                runtime.step(args.tap, 1, full=False)
            terminal = False
            for index in range(args.steps+1):
                start = time.perf_counter()
                state = runtime.snapshot() if index == 0 else (runtime.step_gameplay() if args.start else runtime.step())
                state["sample_ms"] = (time.perf_counter()-start)*1000
                stream.write(json.dumps(state, allow_nan=False)+"\n")
                stream.flush()
                from touhou_ai.telemetry_memory import publish
                publish(packet(state, args.output.stem, "live"))
                if args.start and state.get("transition") == "terminal_or_stage_change":
                    terminal = state["lives_raw"] < 0
                    if not terminal:
                        raise RuntimeError("unverified stage/menu transition")
                    break
            status.update(status="passed", samples=index+1, terminated=terminal,
                          truncated=not terminal, final_frame=state["stage_frame"])
            if args.watchdog_test:
                time.sleep(3.3)
                watchdog = runtime.api.status()
                if watchdog["enabled"] or watchdog["parked"] or watchdog["mask"] != 0 or watchdog["fault"] is None:
                    raise AssertionError("watchdog failed to release game/input")
                if runtime.reader.integer(0x474e30) != 0:
                    raise AssertionError("game input was not neutral after watchdog")
                status["watchdog"] = watchdog
            print(json.dumps({"samples": index+1, "last_frame": state["stage_frame"],
                              "input_calls": state["input_calls"], "output": str(args.output)}))
        except BaseException as error:
            status.update(status="failed", error=str(error))
            raise
        finally:
            runtime.close()
            args.output.with_suffix(".status.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

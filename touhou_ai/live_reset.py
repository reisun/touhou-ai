"""Bounded, observed-menu startup. Never changes stats or skips game initialization."""
import threading


def resume_paused_episode(runtime, budget=120):
    state = runtime.snapshot(full=False)
    def verify(current):
        if ((current['character'], current['shot'], current['difficulty'], current['stage']) != (0, 1, 1, 1)
                or current['mode_flags'] not in (0, 4) or current['replay_mode'] != 0
                or current['player'] is None or current['lives_raw'] < 0):
            raise ValueError('not supported live gameplay')
    verify(state)
    words = state.get('pause_words')
    if words is None or words[:2] != [2, 2] or words[9] != 0 or state['lives_raw'] < 0:
        raise ValueError('requires live paused Resume selection')
    frame = state['stage_frame']
    state = runtime.step(1, 1, full=False)
    for _ in range(budget):
        verify(state)
        if state['pause_words'][1] == 0 and state['stage_frame'] > frame:
            return state
        state = runtime.step(0, 1, full=False)
    raise TimeoutError('paused episode did not resume')


def advance_dialogue(runtime, budget=1800):
    """Confirm dialogue outside policy collection; never navigates a pause/menu."""
    state = runtime.snapshot(full=False)
    for index in range(budget):
        if (not state.get('dialogue_raw') and state.get('player') is not None
                and state.get('mode_flags') in (0, 4) and state.get('replay_mode') == 0):
            return runtime.snapshot()
        if not state.get('dialogue_raw'):
            # A post-boss conversation can end before the next stage's managers exist.
            state = runtime.step(0, 1, full=False)
            continue
        words = state.get('pause_words')
        if (state['player'] is None or state['replay_mode'] != 0 or state['mode_flags'] not in (0, 4)
                or state['lives_raw'] < 0 or words is None or words[1] != 0):
            raise ValueError('dialogue automation requires unpaused live gameplay')
        state = runtime.step(1 if index % 8 == 0 else 0, 1, full=False)
    raise TimeoutError('dialogue did not end within frame budget')


def verify_game_over(state, ready=False):
    if (state.get("lives_raw") != -1 or state.get("player") is None
            or state.get("replay_mode") != 0 or state.get("mode_flags") not in (0, 4)
            or (state.get("character"), state.get("shot"), state.get("difficulty")) != (0, 1, 1)
            or state.get('stage') not in range(1, 7)):
        raise ValueError("not a verified Normal/Reimu B/main-stage game over")
    words = state.get("pause_words")
    if ready and (words is None or words[0:2] != [2, 8] or words[8] != 1
                  or words[11] != 3 or words[9] not in (0, 1, 2)):
        raise ValueError("game-over choices are not ready")


def wait_game_over(runtime, budget=600):
    state = runtime.snapshot(full=False)
    verify_game_over(state)
    # Flush held/pressed/repeat state before interpreting or confirming any menu.
    state = runtime.step(0, 2, full=False)
    dismissed = False
    phases = []
    for _ in range(budget):
        verify_game_over(state)
        words = state.get("pause_words")
        phase = None if words is None else words[1]
        if not phases or phases[-1] != phase:
            phases.append(phase)
        if phase == 12:
            raise RuntimeError(f'unexpected name-entry screen; phases={phases}; words={words}')
        if words is not None and words[0:2] == [2, 8]:
            verify_game_over(state, ready=True)
            return state
        # The Game Over message waits for confirmation before showing its choices.
        if (not dismissed and words is not None and words[0:2] == [2, 6]
                and words[8] == 1 and words[11] in (0, 3) and words[5] >= 120):
            runtime.step(1, 1, full=False)
            dismissed = True
        state = runtime.step(0, 1, full=False)
    raise TimeoutError(f"game-over menu did not become ready; phases={phases}; words={words}")


def continue_episode(runtime, budget=180):
    state = wait_game_over(runtime)
    stage = state['stage']
    # Observed on the pinned executable: Continue / Save replay / Title, index 0..2.
    if stage >= 2:
        return restart_via_title(runtime, state)
    for _ in range(2):
        if state["pause_words"][9] == 0:
            break
        runtime.step(0x10, 1, full=False)
        state = runtime.step(0, 1, full=False)
        verify_game_over(state, ready=True)
    if state["pause_words"][9] != 0:
        raise ValueError("Continue was not selected")
    runtime.step(1, 1, full=False)
    for _ in range(budget):
        state = runtime.step(0, 1, full=False)
        if (state["player"] is not None and state["player"]["status"] == 1
                and state["lives_raw"] == 2 and 0 < state["stage_frame"] <= 180):
            verify_episode(state, stage)
            return state
    raise TimeoutError("Continue did not reset to a verified episode")


def restart_via_title(runtime, state, budget=600):
    """After a later-stage game over, start a new Normal/Reimu B game."""
    verify_game_over(state, ready=True)
    for _ in range(2):
        if state['pause_words'][9] == 2:
            break
        runtime.step(0x20, 1, full=False)
        state = runtime.step(0, 1, full=False)
        verify_game_over(state, ready=True)
    if state['pause_words'][9] != 2:
        raise ValueError('Title was not selected')
    runtime.step(1, 1, full=False)
    for _ in range(budget):
        state = runtime.step(0, 1, full=False)
        words = state.get('menu_words')
        if state.get('player') is None and words is not None:
            if words[7] not in (0, 1, 2):
                raise ValueError('unexpected menu after selecting Title')
            result = start_episode(runtime)
            verify_episode(result, 1)
            return result
        if state.get('player') is not None and state.get('lives_raw', -1) >= 0:
            raise ValueError('Title unexpectedly resumed gameplay')
    raise TimeoutError('Title did not return to the startup menu')


class GameOverHold:
    """Neutral menu frames while CPU training runs; retains the 3-second watchdog."""
    def __init__(self, runtime, publish=None):
        self.runtime, self.publish = runtime, publish
        self.done = threading.Event()
        self.error = None
        self.samples = 0
        self.thread = threading.Thread(target=self._run, name="game-over-hold", daemon=True)

    def start(self):
        self.initial = self.runtime.snapshot(full=False)
        verify_game_over(self.initial, ready=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.done.is_set():
                state = self.runtime.step(0, 2, full=False)
                verify_game_over(state, ready=True)
                if state["stage_frame"] != self.initial["stage_frame"] or state["input_state_raw"][0] != 0:
                    raise RuntimeError("game advanced or non-neutral input during optimizer")
                self.samples += 1
                if self.publish is not None and self.samples % 15 == 0:
                    self.publish(state)
        except BaseException as error:
            self.error = error

    def stop(self):
        self.done.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise TimeoutError("game-over hold failed to stop")
        if self.error is not None:
            raise RuntimeError("game-over hold lost its verified state") from self.error


def menu_command(state):
    words = state.get("menu_words")
    if state.get("player") is not None or words is None:
        raise ValueError("expected a menu, not an active game")
    screen, phase, selection, count = words[7], words[8], words[9], words[11]
    if screen == 0:
        return 0
    if screen not in {1, 2, 6, 7, 8}:
        raise ValueError(f"unsupported menu screen {screen}")
    if phase != 2:
        return 0
    if screen == 1:
        return 1
    target, expected_count = {2: (0, 8), 6: (1, 4), 7: (0, 2), 8: (1, 3)}[screen]
    if count != expected_count or not 0 <= selection < count:
        raise ValueError("menu layout mismatch")
    if selection == target:
        return 1
    return (0x40 if selection > target else 0x80) if screen == 7 else (0x10 if selection > target else 0x20)


def verify_episode(state, stage=1):
    if stage not in range(1, 7) or (state["character"], state["shot"], state["difficulty"], state["stage"]) != (0, 1, 1, stage):
        raise ValueError("reset did not enter Normal/Reimu B/expected stage")
    if state["mode_flags"] not in (0, 4) or state["replay_mode"] != 0 or state["player"] is None:
        raise ValueError("practice/replay/demo/unknown state is not a training episode")
    if not 0 <= state["stage_frame"] <= 180 or state["lives_raw"] != 2:
        raise ValueError("unexpected initial frame/lives")


def start_episode(runtime, budget=900):
    state = runtime.snapshot(full=False)
    if state["player"] is not None:
        raise ValueError("active game/demo; use managed restart instead of overwriting it")
    for _ in range(budget // 2):
        if (state["player"] is not None and state["menu_words"] is None
                and state["player"]["status"] == 1 and state["stage_frame"] > 0):
            verify_episode(state)
            return state
        command = 0 if state["player"] is not None or state["menu_words"] is None else menu_command(state)
        runtime.step(command, 1, full=False)
        state = runtime.step(0, 1, full=False)
    raise TimeoutError("menu startup exceeded frame budget")

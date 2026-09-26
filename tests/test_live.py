import struct
import unittest

from touhou_ai.live_reset import (menu_command, start_episode, verify_episode,
                                  continue_episode, verify_game_over, wait_game_over, GameOverHold)
from touhou_ai.th10_reader import Th10Reader


class Memory:
    def __init__(self):
        self.regions = {0x474c48: bytearray(0x5c)}
        self.regions[0x474e30] = bytearray(10)
        for pointer in (0x47770c, 0x4776ec, 0x4776f4, 0x477830, 0x477834, 0x477838, 0x47784c, 0x4776f0, 0x477818, 0x477704, 0x47781c):
            self.regions[pointer] = bytearray(4)

    def put_pointer(self, address, value):
        self.regions[address] = bytearray(struct.pack("<I", value))

    def read(self, address, size):
        if size > 16384:
            raise ValueError("oversized read")
        for base, data in self.regions.items():
            if base <= address and address+size <= base+len(data):
                return bytes(data[address-base:address-base+size])
        raise OSError("unmapped memory")


class ReaderTests(unittest.TestCase):
    def test_unavailable_not_empty(self):
        state = Th10Reader(Memory()).snapshot()
        self.assertIsNone(state["player"])
        self.assertIsNone(state["bullets"])
        self.assertFalse(state["stable_entity_ids"])

    def test_bullet_pool_excludes_sentinel_and_reads_status(self):
        memory = Memory()
        memory.put_pointer(0x4776f0, 0x1000000)
        pool = bytearray(0x7f0*2001)
        struct.pack_into("<h", pool, 0x446, 1)
        struct.pack_into("<I", pool, 0, 6)
        struct.pack_into("<ff", pool, 0x3b4, -100, 150)
        struct.pack_into("<ff", pool, 0x3c0, 2, 3)
        struct.pack_into("<h", pool, 2000*0x7f0+0x446, 5)
        memory.regions[0x1000060] = pool
        state = Th10Reader(memory).snapshot()
        self.assertEqual(len(state["bullets"]), 1)
        self.assertEqual(state["bullets"][0]["position"], [-100, 150])
        self.assertEqual(state["bullets"][0]["flags_raw"], 6)
        self.assertNotIn("id", state["bullets"][0])

    def test_cycle_and_bad_float_fail_closed(self):
        memory = Memory()
        memory.put_pointer(0x47781c, 0x2000000)
        memory.put_pointer(0x2000018, 0x2001000)
        laser = bytearray(0x48)
        struct.pack_into("<I", laser, 8, 0x2001000)
        memory.regions[0x2001000] = laser
        with self.assertRaisesRegex(ValueError, "cyclic"):
            Th10Reader(memory).snapshot()
        struct.pack_into("<f", laser, 0x24, float("nan"))
        with self.assertRaisesRegex(ValueError, "float"):
            Th10Reader(memory).snapshot()


def menu(screen, phase=2, selection=0, count=8):
    words = [0]*16
    words[7:12] = [screen, phase, selection, selection, count]
    return {"menu_words": words, "player": None}


def gameplay():
    return {"menu_words": None, "player": {"status": 1}, "character": 0, "shot": 1,
            "difficulty": 1, "stage": 1, "stage_frame": 2, "lives_raw": 2,
            "mode_flags": 0, "replay_mode": 0}


class ResetTests(unittest.TestCase):
    def game_over(self, selection=2):
        words = [2, 8, 0, 0, 100, 101, 0, 0, 1, selection, selection, 3]
        return gameplay() | {"lives_raw": -1, "stage_frame": 2000,
                             "pause_words": words, "input_state_raw": [0]*5}

    def test_continue_selects_observed_choice_then_verifies_new_episode(self):
        state = self.game_over()
        class Fake:
            commands = []
            current = state
            def snapshot(self, full=False):
                return self.current
            def step(self, mask, frames, full=False):
                self.commands.append(mask)
                if mask == 0x10:
                    self.current["pause_words"][9] -= 1
                elif mask == 1:
                    self.current = gameplay()
                return self.current
        runtime = Fake()
        self.assertEqual(continue_episode(runtime), gameplay())
        self.assertEqual(runtime.commands, [0, 0x10, 0, 0x10, 0, 1, 0])

    def test_game_over_rejects_active_game_wrong_mode_and_menu(self):
        for state in (gameplay(), self.game_over() | {"mode_flags": 1},
                      self.game_over() | {"pause_words": None},
                      self.game_over() | {"stage": 7}):
            with self.assertRaises(ValueError):
                verify_game_over(state, ready=True)

    def test_main_stage_game_over_supported(self):
        verify_game_over(self.game_over() | {'stage': 2}, ready=True)

    def test_game_over_message_is_dismissed_once_before_choices(self):
        ready = self.game_over()
        message = self.game_over()
        message["pause_words"][1] = 6
        message["pause_words"][5] = 120
        message["pause_words"][11] = 0
        class Fake:
            current = message
            commands = []
            def snapshot(self, full=False):
                return self.current
            def step(self, mask, frames, full=False):
                self.commands.append(mask)
                if mask == 1:
                    self.current = ready
                return self.current
        runtime = Fake()
        self.assertEqual(wait_game_over(runtime), ready)
        self.assertEqual(runtime.commands, [0, 1, 0])

    def test_optimizer_hold_sends_only_neutral_and_detects_game_advance(self):
        import threading
        state = self.game_over()
        sampled = threading.Event()
        class Fake:
            commands = []
            def snapshot(self, full=False):
                return state
            def step(self, mask, frames, full=False):
                self.commands.append(mask)
                sampled.set()
                return state | {"stage_frame": state["stage_frame"]+1}
        runtime = Fake()
        hold = GameOverHold(runtime)
        hold.start()
        self.assertTrue(sampled.wait(1))
        with self.assertRaisesRegex(RuntimeError, "lost"):
            hold.stop()
        self.assertEqual(runtime.commands, [0])

    def test_menu_navigation_and_unknown_rejection(self):
        self.assertEqual(menu_command(menu(0)), 0)
        self.assertEqual(menu_command(menu(1, phase=1)), 0)
        self.assertEqual(menu_command(menu(2)), 1)
        self.assertEqual(menu_command(menu(6, selection=0, count=4)), 0x20)
        self.assertEqual(menu_command(menu(7, selection=1, count=2)), 0x40)
        self.assertEqual(menu_command(menu(8, selection=0, count=3)), 0x20)
        with self.assertRaises(ValueError):
            menu_command(menu(10))
        with self.assertRaises(ValueError):
            menu_command(menu(2, count=9))

    def test_episode_replay_and_wrong_conditions_rejected(self):
        state = gameplay()
        verify_episode(state)
        for key, value in (("replay_mode", 1), ("mode_flags", 32), ("difficulty", 3),
                           ("shot", 0), ("lives_raw", 9), ("stage_frame", 200)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_episode(state | {key: value})

    def test_loading_wait_and_bounded_timeout(self):
        class Fake:
            calls = 0
            def snapshot(self, full=False):
                return menu(0)
            def step(self, mask, frames, full=False):
                self.calls += 1
                return menu(0) if self.calls < 4 else gameplay()
        runtime = Fake()
        self.assertEqual(start_episode(runtime), gameplay())
        self.assertEqual(runtime.calls, 4)
        with self.assertRaises(TimeoutError):
            start_episode(Fake(), budget=2)


class FrameTests(unittest.TestCase):
    def test_action_mapping_and_validation(self):
        from touhou_ai.live_runtime import input_mask
        self.assertEqual(input_mask([2, 1, 1, 1]), 0x97)
        self.assertEqual(input_mask([0, 0, 0, 0]), 0)
        for action in ([9, 0, 0, 0], [0.5, 0, 0, 0], [True, 0, 0, 0], []):
            with self.assertRaises(ValueError):
                input_mask(action)
    def test_stage_frame_mismatch_is_not_silently_accepted(self):
        from touhou_ai.live_runtime import LiveRuntime
        class Fake:
            snapshot = lambda self, full=False: gameplay()
            step = lambda self, mask, frames, full, **kwargs: gameplay() | {"stage_frame": 5}
        with self.assertRaisesRegex(RuntimeError, "frame mismatch"):
            LiveRuntime.step_gameplay(Fake())

    def test_exact_frames_and_terminal_boundary(self):
        from touhou_ai.live_runtime import LiveRuntime
        class Fake:
            snapshot = lambda self, full=False: gameplay()
            def step(self, mask, frames, full, gameplay_guard=False):
                if not gameplay_guard:
                    raise AssertionError('policy steps require terminal input guard')
                return gameplay() | {"stage_frame": 4}
        self.assertEqual(LiveRuntime.step_gameplay(Fake())["transition"], "gameplay")
        Fake.step = lambda self, mask, frames, full, **kwargs: gameplay() | {"lives_raw": -1}
        self.assertEqual(LiveRuntime.step_gameplay(Fake())["transition"], "terminal_or_stage_change")

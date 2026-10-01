import copy
import unittest

from touhou_ai.live_transition import bridge_transition, CollectionInterrupted


def state(**changes):
    return dict(character=0, shot=1, difficulty=1, stage=1, stage_frame=8800,
                lives_raw=1, mode_flags=0, replay_mode=0, player={'status': 1},
                dialogue_raw=0, pause_words=[2, 0]) | changes


class Runtime:
    def __init__(self, states):
        self.states, self.index, self.commands = states, 0, []

    def snapshot(self, full=True):
        return copy.deepcopy(self.states[self.index])

    def step(self, mask, frames, full=False):
        self.commands.append((mask, frames))
        self.index = min(self.index+1, len(self.states)-1)
        return self.snapshot(full)


class TransitionTests(unittest.TestCase):
    def test_loading_with_score_flag_keeps_loading_guards(self):
        for flags in (2, 6):
            with self.subTest(flags=flags):
                score = flags & 4
                frames = [state(stage=3, mode_flags=flags, screen_state_raw=11,
                                dialogue_raw=123),
                          state(stage=4, stage_frame=0, mode_flags=flags,
                                screen_state_raw=7, player=None, replay_mode=None),
                          state(stage=4, stage_frame=1, mode_flags=score,
                                screen_state_raw=7)]
                runtime = Runtime(frames)
                result, info = bridge_transition(runtime, runtime.snapshot(), 3)
                self.assertEqual((result['stage'], result['mode_flags']), (4, score))
                self.assertEqual(runtime.commands, [(0, 1), (0, 1)])
                self.assertEqual(info['loading_frames'], 2)
                for changes in [dict(screen_state_raw=7), dict(replay_mode=1),
                                dict(pause_words=[2, 2]), dict(stage=5),
                                dict(mode_flags=flags | 1), dict(mode_flags=flags | 8)]:
                    runtime = Runtime([frames[0] | changes])
                    with self.assertRaises(ValueError):
                        bridge_transition(runtime, runtime.snapshot(), 3)
                    self.assertEqual(runtime.commands, [])
                for changes in [dict(stage=3), dict(stage_frame=1), dict(stage=5)]:
                    runtime = Runtime([frames[1] | changes])
                    with self.assertRaises(ValueError):
                        bridge_transition(runtime, runtime.snapshot(), 3)
                    self.assertEqual(runtime.commands, [])
                runtime = Runtime([frames[0] | dict(stage=6)])
                with self.assertRaises(ValueError):
                    bridge_transition(runtime, runtime.snapshot(), 6)
                self.assertEqual(runtime.commands, [])

    def test_recorded_stage_three_score_flag(self):
        captured = state(stage=3, stage_frame=3745, mode_flags=4,
                         screen_state_raw=7, stage_init_pending=False)
        runtime = Runtime([captured])
        result, info = bridge_transition(runtime, runtime.snapshot(), 3)
        self.assertEqual(result, captured | {'progress_reward_events': []})
        self.assertEqual(runtime.commands, [])
        self.assertEqual(info['automatic_frames'], 0)
        from touhou_ai.live_learning import hit_events
        self.assertEqual(len(hit_events(captured, captured | {'lives_raw': 0})), 1)
        from touhou_ai.live_reset import verify_game_over
        verify_game_over(captured | {'lives_raw': -1})
        from touhou_ai.live_runtime import LiveRuntime
        from unittest.mock import Mock
        runtime = Mock()
        runtime.snapshot.return_value = captured
        runtime.step.return_value = captured | {'stage_frame': 3747}
        self.assertEqual(LiveRuntime.step_gameplay(runtime)['transition'], 'gameplay')
        for changes in [dict(replay_mode=1), dict(pause_words=[2, 2]),
                        dict(stage=5), dict(mode_flags=5), dict(mode_flags=6),
                        dict(mode_flags=12), dict(mode_flags=36)]:
            runtime = Runtime([captured | changes])
            with self.assertRaises(ValueError):
                bridge_transition(runtime, runtime.snapshot(), 3)
            self.assertEqual(runtime.commands, [])

    def test_incoming_stage_clock_reset_is_not_a_policy_step(self):
        frames = [state(stage=2, stage_frame=0, mode_flags=2, screen_state_raw=7)]
        frames += [state(stage=2, stage_frame=frame, screen_state_raw=7,
                         stage_init_pending=True, stage_manager_raw={'flags': 0x800, 'timer': frame},
                         dialogue_raw=123 if frame == 8 else 0)
                   for frame in range(1, 31)]
        frames += [state(stage=2, stage_frame=1, screen_state_raw=7,
                         stage_init_pending=False, stage_manager_raw={'flags': 0, 'timer': 31})]
        runtime = Runtime(frames)
        after, info = bridge_transition(runtime, runtime.snapshot(), 1)
        self.assertEqual(after['stage_frame'], 1)
        self.assertFalse(after['stage_init_pending'])
        self.assertEqual(runtime.commands, [(0, 1)] * 31)
        self.assertEqual(info['loading_frames'], 31)
        # Also reject an initialization snapshot in the ordinary fast path.
        from unittest.mock import Mock
        runtime = Mock()
        runtime.step.return_value = frames[-1]
        runtime.snapshot.return_value = frames[-1]
        initial = frames[20] | dict(snapshot_full=True, frame_locked=True)
        after, _ = bridge_transition(runtime, initial, 2)
        runtime.step.assert_called_once_with(0, 1, full=False)
        self.assertFalse(after['stage_init_pending'])

    def test_recorded_stage_two_initialization_with_screen_already_seven(self):
        # Exact guard fields from the failure at full-policy update 156.
        captured = state(stage=2, stage_frame=0, mode_flags=2, screen_state_raw=7,
                         pause_words=[2, 0, 110673432, 110672664, 7719, 7720,
                                      1173438464, 4681592, 1, 0, 0, 0])
        runtime = Runtime([captured, captured, state(stage=2, stage_frame=1, screen_state_raw=7)])
        result, info = bridge_transition(runtime, runtime.snapshot(), 1)
        self.assertEqual((result['stage'], result['stage_frame']), (2, 1))
        self.assertEqual(runtime.commands, [(0, 1), (0, 1)])
        self.assertEqual(info['loading_frames'], 2)
        for changes in [dict(stage=1), dict(stage=3), dict(stage_frame=1),
                        dict(replay_mode=1), dict(pause_words=[2, 2]), dict(mode_flags=3)]:
            runtime = Runtime([captured | changes])
            with self.assertRaises(ValueError):
                bridge_transition(runtime, runtime.snapshot(), 1)
            self.assertEqual(runtime.commands, [])

    def test_native_next_stage_loading_is_waited_with_neutral_input(self):
        runtime = Runtime([state(mode_flags=2, screen_state_raw=11, dialogue_raw=123),
                           state(stage=2, stage_frame=0, mode_flags=2, screen_state_raw=11,
                                 player=None, replay_mode=None),
                           state(stage=2, stage_frame=1, screen_state_raw=7)])
        after, info = bridge_transition(runtime, runtime.snapshot(), 1)
        self.assertEqual(after['stage'], 2)
        self.assertEqual(runtime.commands, [(0, 1), (0, 1)])
        self.assertEqual(info['loading_frames'], 2)
        for changes in [dict(mode_flags=2, screen_state_raw=7),
                        dict(mode_flags=3, screen_state_raw=11),
                        dict(mode_flags=2, screen_state_raw=11, replay_mode=1),
                        dict(mode_flags=2, screen_state_raw=11, pause_words=[2, 2])]:
            runtime = Runtime([state(**changes)])
            with self.assertRaises(ValueError):
                bridge_transition(runtime, runtime.snapshot(), 1)
            self.assertEqual(runtime.commands, [])

    def test_reuse_only_current_full_normal_observation(self):
        from unittest.mock import Mock
        raw = state(snapshot_full=True, frame_locked=True, gate_tick=42,
                    combat_reward_events=[{'id':'damage'}], progress_reward_events=[{'id':'spell'}])
        runtime = Mock()
        runtime.owns_full_snapshot.return_value = True
        result, info = bridge_transition(runtime, raw, 1)
        self.assertIs(result, raw)
        runtime.snapshot.assert_not_called()
        runtime.step.assert_not_called()
        self.assertEqual(info['automatic_frames'], 0)
        for changes in ({'snapshot_full':False}, {'stage':2}, {'lives_raw':-1}):
            runtime.reset_mock()
            current=raw | changes
            runtime.snapshot.return_value=current
            bridge_transition(runtime,current,1)
            runtime.snapshot.assert_called_once()
        runtime.reset_mock()
        runtime.owns_full_snapshot.return_value=False
        runtime.snapshot.return_value=raw
        bridge_transition(runtime,raw,1)
        runtime.snapshot.assert_called_once()

    def test_full_snapshot_ownership_requires_parked_matching_gate(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from touhou_ai.live_runtime import LiveRuntime
        runtime=SimpleNamespace(park={'tick':42},api=Mock())
        raw=state(snapshot_full=True,frame_locked=True,gate_tick=42)
        runtime.api.status.return_value={'enabled':True,'parked':True,'tick':42}
        self.assertTrue(LiveRuntime.owns_full_snapshot(runtime,raw))
        for changes in ({'snapshot_full':False},{'frame_locked':False},{'gate_tick':41}):
            self.assertFalse(LiveRuntime.owns_full_snapshot(runtime,raw | changes))
        for changes in ({'enabled':False},{'parked':False},{'tick':43}):
            runtime.api.status.return_value={'enabled':True,'parked':True,'tick':42} | changes
            self.assertFalse(LiveRuntime.owns_full_snapshot(runtime,raw))

    def test_progress_events_survive_dialogue_loading_and_snapshot_duplicates(self):
        a, b, c = ({'id': name, 'kind': 'progress'} for name in ('arrival', 'defeat', 'next-arrival'))
        runtime = Runtime([state(dialogue_raw=123, progress_reward_events=[a]),
                           state(player=None, progress_reward_events=[b]),
                           state(stage=2, stage_frame=1, progress_reward_events=[c])])
        after, _ = bridge_transition(runtime, runtime.snapshot(), 1)
        self.assertEqual(after['progress_reward_events'], [a, b, c])

    def test_dialogue_loading_and_new_stage_are_one_continuation(self):
        runtime = Runtime([state(dialogue_raw=123),
                           state(player=None, replay_mode=None),
                           state(stage=2, stage_frame=0, player=None, replay_mode=None),
                           state(stage=2, stage_frame=1)])
        after, info = bridge_transition(runtime, runtime.snapshot(), 1)
        self.assertEqual(after['stage'], 2)
        self.assertEqual(after['lives_raw'], 1)
        self.assertEqual(runtime.commands, [(1, 1), (0, 1), (0, 1)])
        self.assertEqual(info, dict(from_stage=1, to_stage=2, automatic_frames=3,
                                   dialogue_frames=1, loading_frames=2))
        self.assertFalse(after['lives_raw'] < 0)

    def test_first_playable_frame_returns_without_extra_neutral_frame(self):
        runtime = Runtime([state(stage=2, stage_frame=1)])
        after, info = bridge_transition(runtime, runtime.snapshot(), 1)
        self.assertEqual(info['automatic_frames'], 0)
        self.assertEqual(runtime.commands, [])
        self.assertEqual(after['stage_frame'], 1)

    def test_game_over_remains_terminal_not_a_restart(self):
        runtime = Runtime([state(lives_raw=-1, pause_words=[2, 6])])
        after, _ = bridge_transition(runtime, runtime.snapshot(), 1)
        self.assertEqual(after['lives_raw'], -1)
        self.assertEqual(runtime.commands, [])

    def test_pause_replay_stage_skip_and_wrong_mode_are_not_dismissed(self):
        for changes in [dict(pause_words=[2, 2]), dict(replay_mode=1), dict(mode_flags=1),
                        dict(stage=3), dict(stage=0), dict(character=1)]:
            runtime = Runtime([state(**changes)])
            with self.assertRaises(ValueError):
                bridge_transition(runtime, runtime.snapshot(), 1)
            self.assertEqual(runtime.commands, [])

    def test_loading_stop_and_timeout_are_bounded(self):
        runtime = Runtime([state(player=None, replay_mode=None)])
        with self.assertRaises(CollectionInterrupted):
            bridge_transition(runtime, runtime.snapshot(), 1, should_stop=lambda: True)
        self.assertEqual(runtime.commands, [])
        with self.assertRaises(TimeoutError):
            bridge_transition(runtime, runtime.snapshot(), 1, budget=3)
        self.assertEqual(runtime.commands, [(0, 1)]*3)

    def test_boundary_is_not_a_hit_and_event_ids_include_stage(self):
        from touhou_ai.live_learning import hit_events
        from touhou_ai.live_features import bomb_events
        self.assertEqual(hit_events(state(), state(stage=2, stage_frame=1), True), [])
        a = hit_events(state(), state(stage_frame=10, lives_raw=0))[0]
        b = hit_events(state(stage=2), state(stage=2, stage_frame=10, lives_raw=0))[0]
        self.assertNotEqual(a['id'], b['id'])
        a = bomb_events(state(bomb={'state': 0}), state(bomb={'state': 1}))[0]
        b = bomb_events(state(stage=2, bomb={'state': 0}), state(stage=2, bomb={'state': 1}))[0]
        self.assertNotEqual(a['id'], b['id'])

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

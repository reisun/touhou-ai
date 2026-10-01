import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from touhou_ai.live_recovery import recovery_allowed
from touhou_ai.live_transition import CollectionInterrupted


class RecoveryTests(unittest.TestCase):
    def test_user_stop_pause_and_retry_limit_win(self):
        runtime = Mock()
        runtime.api.status.return_value = {'fault': None}
        runtime.snapshot.return_value = {'lives_raw': 2, 'pause_words': [0, 0]}
        error = TimeoutError('unknown screen')
        self.assertTrue(recovery_allowed(runtime, error, False, 0))
        self.assertFalse(recovery_allowed(runtime, error, True, 0))
        self.assertFalse(recovery_allowed(runtime, error, False, 2))
        self.assertFalse(recovery_allowed(runtime, CollectionInterrupted(), False, 0))
        self.assertFalse(recovery_allowed(None, error, False, 0))
        runtime.api.status.return_value = {'fault': 'emergency F8'}
        self.assertFalse(recovery_allowed(runtime, error, False, 0))
        runtime.api.status.return_value = {'fault': None}
        runtime.snapshot.return_value = {'lives_raw': 2, 'pause_words': [0, 2]}
        self.assertFalse(recovery_allowed(runtime, error, False, 0))

    def test_dead_process_can_recover(self):
        runtime = Mock()
        runtime.api.status.side_effect = RuntimeError('process gone')
        runtime.snapshot.side_effect = OSError('process gone')
        self.assertTrue(recovery_allowed(runtime, OSError('read failed'), False, 0))

    def test_unlimited_recovery_preserves_stop_and_ownership_guards(self):
        runtime = Mock()
        runtime.api.status.return_value = {'fault': None}
        runtime.snapshot.return_value = {'lives_raw': 2, 'pause_words': [0, 0]}
        error = TimeoutError('unknown screen')
        self.assertTrue(recovery_allowed(runtime, error, False, 10000, max_attempts=None))
        self.assertFalse(recovery_allowed(runtime, error, True, 10000, max_attempts=None))
        self.assertFalse(recovery_allowed(None, error, False, 10000, max_attempts=None))
        self.assertFalse(recovery_allowed(runtime, CollectionInterrupted(), False, 10000, max_attempts=None))
        runtime.api.status.return_value = {'fault': 'emergency F8'}
        self.assertFalse(recovery_allowed(runtime, error, False, 10000, max_attempts=None))
        runtime.api.status.return_value = {'fault': None}
        runtime.snapshot.return_value = {'lives_raw': 2, 'pause_words': [0, 2]}
        self.assertFalse(recovery_allowed(runtime, error, False, 10000, max_attempts=None))

    def test_collector_fault_updates_valid_prefix_then_restarts_stage_one(self):
        from touhou_ai import live_learning as learner
        from tests.test_live_learning import LiveLearningTests

        for valid_steps in (4, 32):
            with self.subTest(valid_steps=valid_steps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'configs').mkdir()
                shutil.copy(learner.ROOT / 'configs/sharu-inspired-v1.json', root / 'configs')
                (root / '.runtime').mkdir()
                (root / '.runtime/game.json').write_text(json.dumps({'Id': 123}))
                state = LiveLearningTests().state() | {'pause_words': [0, 0]}
                runtime = Mock()
                runtime.api.status.return_value = {'fault': None}
                runtime.snapshot.return_value = state
                calls = []

                def step(mask, frames):
                    if len(calls) == valid_steps:
                        raise TimeoutError('injected crash, no actual game involved')
                    calls.append(mask)
                    return state | {'stage_frame': len(calls)*2, 'input_state_raw': [mask]}

                runtime.step_gameplay.side_effect = step
                restarted = Mock()
                restarted.snapshot.return_value = state
                with patch.object(learner, 'ROOT', root), \
                     patch.object(learner, 'LiveRuntime', side_effect=[runtime, restarted]), \
                     patch.object(learner, 'game_command') as command, \
                     patch.object(learner, 'start_episode') as start, \
                     patch.object(learner, 'bridge_transition', side_effect=lambda rt, s, *a, **k: (s, {'automatic_frames': 0})), \
                     patch.object(learner, 'packet', return_value={'capabilities': {}}), \
                     patch.object(learner, 'publish_telemetry', return_value=True), \
                     patch('touhou_ai.model_monitor.model_metadata', return_value={}), \
                     patch('touhou_ai.live_acceptance.pause', return_value={'pause_words': [0, 2]}):
                    report = learner.train(root / 'artifacts/test', episodes=1, max_steps=40)
                self.assertEqual([c.args[0] for c in command.call_args_list], ['start', 'stop', 'start'])
                self.assertTrue(command.call_args_list[1].kwargs['recovery'])
                self.assertEqual(start.call_count, 2)
                self.assertFalse(report['detailed_logs'])
                self.assertEqual(list((root/'artifacts/test').glob('episode-*.jsonl')), [])
                self.assertTrue(report['recoveries'][0]['restarted'])
                self.assertTrue(report['paused_on_exit'])
                self.assertEqual(len(report['episodes']), int(valid_steps >= 32))
                if valid_steps >= 32:
                    entry = report['episodes'][0]
                    self.assertEqual(entry['steps'], 32)
                    self.assertTrue(entry['truncated'])
                    self.assertFalse(entry['terminated'])
                    self.assertTrue(entry['reload_verified'])
                    self.assertTrue(entry['recovery_update'])
                    self.assertFalse(entry['updated_at_game_over'])
                else:
                    self.assertEqual(report['gameplay_training_steps'], 0)

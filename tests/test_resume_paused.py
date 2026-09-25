import unittest
from touhou_ai.live_reset import resume_paused_episode


class ResumePausedTests(unittest.TestCase):
    def test_midplay_resume_and_reject_wrong_menu(self):
        class Runtime:
            def __init__(self):
                self.inputs = []
                self.state = dict(character=0, shot=1, difficulty=1, stage=1, mode_flags=0,
                                  replay_mode=0, player={}, lives_raw=1, stage_frame=7904,
                                  pause_words=[2, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3])
            def snapshot(self, full=False):
                return self.state
            def step(self, mask, frames, full=False):
                self.inputs.append(mask)
                self.state = self.state | {'stage_frame': 7905, 'pause_words': [2, 0]}
                return self.state
        runtime = Runtime()
        self.assertEqual(resume_paused_episode(runtime)['stage_frame'], 7905)
        self.assertEqual(runtime.inputs, [1])
        runtime = Runtime()
        runtime.state['pause_words'][9] = 1
        with self.assertRaises(ValueError):
            resume_paused_episode(runtime)
        self.assertEqual(runtime.inputs, [])

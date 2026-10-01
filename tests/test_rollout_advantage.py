import unittest
from unittest.mock import patch
import numpy as np
from touhou_ai.rollout_advantage import build
from touhou_ai.separate_clip_ppo import SeparateClipPPO

class RolloutAdvantageTest(unittest.TestCase):
    def test_global_normalization_and_restoration(self):
        m,_,_=build(7);b=m.rollout_buffer
        original=np.array([[-4.],[-1.],[0.],[2.]],np.float32);b.advantages=original
        expected=(original-original.mean())/(original.std(ddof=1)+1e-8)
        def check(instance):
            self.assertFalse(instance.normalize_advantage)
            np.testing.assert_allclose(b.advantages,expected)
        with patch.object(SeparateClipPPO,'train',check):m.train()
        self.assertIs(b.advantages,original);self.assertTrue(m.normalize_advantage)
        with patch.object(SeparateClipPPO,'train',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):m.train()
        self.assertIs(b.advantages,original);self.assertTrue(m.normalize_advantage)

if __name__=='__main__':unittest.main()

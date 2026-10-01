import json,random,tempfile,unittest
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from tests.test_directml_update import TinyEnv
from touhou_ai.checkpoint_rng import load_preserving_rng,capture_rng,restore_rng

class CheckpointRngTests(unittest.TestCase):
    def test_loading_does_not_change_action_draws_or_shuffle(self):
        torch.set_num_threads(1)
        m=PPO('MultiInputPolicy',TinyEnv(),n_steps=8,batch_size=4,seed=7)
        obs={'player':np.ones(4,dtype=np.float32)}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model'
            m.collector_rng_state=capture_rng();m.save(path)
            torch.manual_seed(91);np.random.seed(92);random.seed(93)
            state=capture_rng()
            expected=[m.predict(obs)[0].tolist() for _ in range(32)]
            shuffle=np.random.permutation(32);py=random.random()
            restore_rng(state)
            loaded=load_preserving_rng(PPO,path,device='cpu')
            self.assertEqual(loaded.collector_rng_state,m.collector_rng_state)
            actual=[loaded.predict(obs)[0].tolist() for _ in range(32)]
            self.assertEqual(expected,actual)
            np.testing.assert_array_equal(shuffle,np.random.permutation(32));self.assertEqual(py,random.random())
            restore_rng(json.loads(json.dumps(state)));self.assertEqual(capture_rng(),state)

    def test_failed_load_restores_state(self):
        state=capture_rng()
        class Broken:
            @staticmethod
            def load(*args,**kwargs):
                random.seed(3);np.random.seed(3);torch.manual_seed(3);raise RuntimeError('failed load')
        with self.assertRaises(RuntimeError):load_preserving_rng(Broken,'missing')
        self.assertEqual(capture_rng(),state)

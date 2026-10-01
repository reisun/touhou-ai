import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from touhou_ai.autumn_training import build_model,build_reference_model
from touhou_ai.spatial_input_candidates import CandidateFeatures,risk_grid
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import distributions

class AdoptedSpatialInputTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_adopted_factory_learning_and_resume(self):
        model,cls,config=build_model('cnn',7)
        self.assertEqual(config['version'],'autumn-baseline-v4')
        self.assertEqual(config['reward_death'],-1)
        self.assertIsInstance(model,SeparateClipPPO)
        self.assertIsInstance(model.policy.pi_features_extractor,CandidateFeatures)
        self.assertEqual(model.observation_space['action_grid'].shape,(4,3,3))
        self.assertEqual(model.observation_space['local_grid'].shape,(6,96,96))
        np.testing.assert_array_equal(model.action_space.nvec,[9,2,2,2])
        self.assertIsInstance(model.policy.mlp_extractor.value_net[-1],torch.nn.Tanh)
        model.n_steps=16;model.batch_size=8;model.n_epochs=1
        model.rollout_buffer=GridRolloutBuffer(16,model.observation_space,model.action_space,device='cpu',gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=1)
        before={k:v.clone() for k,v in model.policy.state_dict().items()}
        model.learn(32)
        self.assertTrue(any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items()))
        obs=cls().reset(seed=81)[0]
        with torch.no_grad():
            ds=distributions(model.policy,obs)
            for head in (1,3):self.assertEqual(float(ds[head].probs[0,0]),1.)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model';model.save(path)
            loaded=SeparateClipPPO.load(path,env=cls(),device='cpu')
            for k,v in model.policy.state_dict().items():torch.testing.assert_close(v,loaded.policy.state_dict()[k],rtol=0,atol=0)
            loaded.learn(16,reset_num_timesteps=False)
            self.assertEqual(loaded.num_timesteps,48)

    def test_reference_contract_and_unsupported_lasers(self):
        model,_,config=build_reference_model('cnn',7)
        self.assertEqual(config['version'],'autumn-baseline-v2')
        self.assertNotIn('action_grid',model.observation_space.spaces)
        _,cls,_=build_model('cnn',7)
        env=cls();env.reset(seed=7);raw=env.raw_observation();raw['lasers']=[{}]
        with self.assertRaisesRegex(ValueError,'bullet-only'):risk_grid(raw)

if __name__=='__main__':unittest.main()

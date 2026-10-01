import inspect
import unittest
import tempfile
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import SpeedPolicy, NumericalAblation, GridAblation
from touhou_ai.narrow_grid import NarrowGridFeatures

class SeparateClipTests(unittest.TestCase):
    def test_only_train_change_is_clipping(self):
        expected=inspect.getsource(PPO.train).replace('th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)', 'self.clip_separate_gradients()')
        self.assertEqual(inspect.getsource(SeparateClipPPO.train), expected)

    def test_actor_clip_independent_of_critic_and_parameters_exhaustive(self):
        torch.set_num_threads(1)
        for cnn in [False, True]:
            kw={'share_features_extractor':False,'net_arch':{'pi':[16],'vf':[16]}}
            if cnn:kw['features_extractor_class']=NarrowGridFeatures
            m=SeparateClipPPO(SpeedPolicy, GridAblation() if cnn else NumericalAblation(relative=True), policy_kwargs=kw,n_steps=8,batch_size=8,seed=7)
            m.set_logger(configure(format_strings=[]))
            p=m.policy
            actor=list(p.pi_features_extractor.parameters())+list(p.mlp_extractor.policy_net.parameters())+list(p.action_net.parameters())
            critic=list(p.vf_features_extractor.parameters())+list(p.mlp_extractor.value_net.parameters())+list(p.value_net.parameters())
            results=[]
            for scale in [1.,1000.]:
                for x in actor:x.grad=torch.ones_like(x)
                for x in critic:x.grad=torch.full_like(x,scale)
                m.clip_separate_gradients()
                results.append(torch.cat([x.grad.flatten() for x in actor]))
                self.assertLessEqual(float(torch.cat([x.grad.flatten() for x in critic]).norm()),.5001)
            torch.testing.assert_close(results[0],results[1])
            self.assertAlmostEqual(float(results[0].norm()),.5,places=4)

    def test_train_save_resume(self):
        torch.set_num_threads(1)
        m=SeparateClipPPO(SpeedPolicy,NumericalAblation(relative=True),n_steps=8,batch_size=8,n_epochs=1,seed=7,policy_kwargs={'share_features_extractor':False,'net_arch':{'pi':[16],'vf':[16]}})
        m.learn(8)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'model'
            m.save(path)
            restored=SeparateClipPPO.load(path,env=NumericalAblation(relative=True))
            for key,value in m.policy.state_dict().items():
                torch.testing.assert_close(restored.policy.state_dict()[key],value)
            restored.learn(8,reset_num_timesteps=False)
            self.assertEqual(restored.num_timesteps,16)
            self.assertIn('train/actor_gradient_norm',restored.logger.name_to_value)

if __name__=='__main__':unittest.main()

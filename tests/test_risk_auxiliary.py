import functools,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from touhou_ai.autumn_training import build_reference_model as build_model
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.risk_auxiliary import RiskTargets,RiskAuxPolicy,WithoutRiskTargets,observed_risk
from touhou_ai.risk_aux_ppo import RiskAuxPPO
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer


def make(rep,coef=.1,balance="scene"):
    cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
    env=RiskTargets(cls)
    kw={'risk_balance':balance,'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]},'features_extractor_class':WithoutRiskTargets if rep=='relative' else NarrowGridFeatures}
    model=RiskAuxPPO(RiskAuxPolicy,env,seed=7,device='cpu',n_steps=8,batch_size=8,n_epochs=1,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,aux_coef=coef,separate_clip=rep=='cnn',rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw)
    return model,env

class RiskAuxTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_geometry_and_reference(self):
        from scripts.audit_observation_credit import linear_prediction,snapshot
        e=NumericalAblation(relative=True);e.reset(seed=7)
        for i in range(100):
            if i%10==0:
                np.testing.assert_array_equal(observed_risk(e.pos,e.xy[e.alive],e.vel[e.alive]),1-linear_prediction(snapshot(e)).astype(np.float32))
            _,_,d,_,_=e.step([0,0,0,0])
            if d:break
        self.assertEqual(observed_risk([0,300],[],[]).sum(),0)
        risks=observed_risk([0,300],[[0,280]],[[0,2]])
        self.assertEqual(risks[0],1)
        self.assertTrue(np.any(risks==0))
    def test_no_label_leak_and_same_initial_weights(self):
        for rep in ['relative','cnn']:
            base,_,_=build_model(rep,7);m,e=make(rep)
            for key,value in base.policy.state_dict().items():torch.testing.assert_close(m.policy.state_dict()[key],value,rtol=0,atol=0)
            o,_=e.reset(seed=7);batch={k:np.stack([v,v]) for k,v in o.items()}
            batch['risk_targets'][0]=0;batch['risk_targets'][1]=1
            t,_=m.policy.obs_to_tensor(batch)
            pi,vf=m.policy._latents(t)
            torch.testing.assert_close(pi[0],pi[1]);torch.testing.assert_close(vf[0],vf[1])
            t['risk_targets'][:,::2]=0;t['risk_targets'][:,1::2]=1
            m.policy.evaluate_actions(t,torch.zeros((2,4)))
            m.policy.optimizer.zero_grad();m.policy.risk_aux_loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.policy.mlp_extractor.policy_net.parameters()))
            self.assertTrue(all(p.grad is None for p in m.policy.mlp_extractor.value_net.parameters()))
            self.assertTrue(all(p.grad is None for p in m.policy.action_net.parameters()))
    def test_zero_coefficient_learning_equivalence_and_resume(self):
        for rep in ['relative','cnn']:
            base,_,_=build_model(rep,7);base.n_steps=8;base.batch_size=8;base.n_epochs=1
            base.rollout_buffer=GridRolloutBuffer(8,base.observation_space,base.action_space,device='cpu',gamma=.9995,gae_lambda=.95,n_envs=1)
            torch.manual_seed(991);np.random.seed(991);base.learn(8)
            m,e=make(rep,coef=0)
            torch.manual_seed(991);np.random.seed(991);m.learn(8)
            for key,value in base.policy.state_dict().items():torch.testing.assert_close(m.policy.state_dict()[key],value,rtol=1e-5,atol=1e-6)
            with tempfile.TemporaryDirectory() as root:
                p=Path(root)/'model';m.save(p);loaded=RiskAuxPPO.load(p,env=e)
                loaded.aux_coef=.1;loaded.learn(8,reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps,16)

    def test_action_balance_rejects_constant_direction_prior(self):
        m,e=make('relative',balance='action');o,_=e.reset(seed=7)
        batch={k:np.stack([v]*10) for k,v in o.items()}
        target=np.zeros((10,18),dtype=np.float32)
        for a in range(18):target[:1+a%8,a]=1
        batch['risk_targets']=target;t,_=m.policy.obs_to_tensor(batch)
        with torch.no_grad():m.policy.risk_head.weight.zero_();m.policy.risk_head.bias.zero_()
        m.policy.evaluate_actions(t,torch.zeros((10,4)));unbiased=float(m.policy.risk_aux_loss.detach())
        with torch.no_grad():m.policy.risk_head.bias.copy_(torch.linspace(-2,2,18))
        m.policy.evaluate_actions(t,torch.zeros((10,4)));biased=float(m.policy.risk_aux_loss.detach())
        self.assertAlmostEqual(unbiased,np.log(2),places=6)
        self.assertGreater(biased,unbiased)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'action';m.save(path);restored=RiskAuxPPO.load(path,env=e)
            self.assertEqual(restored.policy.risk_balance,'action')
            policy_path=Path(root)/'policy.pt';m.policy.save(policy_path)
            policy=RiskAuxPolicy.load(policy_path)
            self.assertEqual(policy.risk_balance,'action')

if __name__=='__main__':unittest.main()

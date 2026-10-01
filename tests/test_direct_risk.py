import functools,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.autumn_training import build_reference_model as build_model
from touhou_ai.direct_risk import DirectRiskInput,DirectRiskPolicy,WithoutActionRisk,ACTION_RISK
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.risk_auxiliary import observed_risk

def make(rep):
    cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
    env=DirectRiskInput(cls)
    kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]},'features_extractor_class':WithoutActionRisk if rep=='relative' else NarrowGridFeatures}
    alg=PPO if rep=='relative' else SeparateClipPPO
    return alg(DirectRiskPolicy,env,seed=7,n_steps=8,batch_size=8,n_epochs=1,learning_rate=.0001,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw),env

class DirectRiskTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_existing_weights_and_initial_actions_preserved(self):
        for rep in ['relative','cnn']:
            base,_,_=build_model(rep,7);m,e=make(rep)
            for key,value in base.policy.state_dict().items():
                actual=m.policy.state_dict()[key]
                if key=='action_net.weight':actual=actual[:,:value.shape[1]]
                torch.testing.assert_close(actual,value,rtol=0,atol=0)
            o,_=e.reset(seed=7);o[ACTION_RISK]=np.ones(18,dtype=np.float32)
            t,_=m.policy.obs_to_tensor(o);b,_=base.policy.obs_to_tensor({k:v for k,v in o.items() if k!=ACTION_RISK})
            for a,c in zip(m.policy.get_distribution(t).distribution,base.policy.get_distribution(b).distribution):torch.testing.assert_close(a.probs,c.probs)
    def test_risk_reaches_action_only_and_has_trainable_gradient(self):
        for rep in ['relative','cnn']:
            m,e=make(rep);o,_=e.reset(seed=7);batch={k:np.stack([v,v]) for k,v in o.items()}
            batch[ACTION_RISK][0]=0;batch[ACTION_RISK][1]=1
            t,_=m.policy.obs_to_tensor(batch)
            a,v=m.policy._latents(t);torch.testing.assert_close(a[0],a[1]);torch.testing.assert_close(v[0],v[1])
            with torch.no_grad():m.policy.action_net.weight[1,-18]=1
            ds=m.policy.get_distribution(t).distribution
            self.assertGreater(float(ds[0].probs[1,1].detach()),float(ds[0].probs[0,1].detach()))
            values,logp,ent=m.policy.evaluate_actions(t,torch.tensor([[1,0,0,0],[1,0,0,0]]))
            m.policy.optimizer.zero_grad();(-logp.mean()).backward()
            self.assertGreater(float(m.policy.action_net.weight.grad[:,-18:].abs().sum()),0)
            torch.testing.assert_close(values[0],values[1])
    def test_wrapped_physics_and_lifecycle(self):
        raw=NumericalAblation(relative=True);wrapped=DirectRiskInput(lambda:NumericalAblation(relative=True))
        a,_=raw.reset(seed=7);b,_=wrapped.reset(seed=7)
        for _ in range(130):
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            np.testing.assert_array_equal(b[ACTION_RISK],observed_risk(raw.pos,raw.xy[raw.alive],raw.vel[raw.alive]))
            a,r,d,t,i=raw.step([0,0,0,0]);b,s,e,u,j=wrapped.step([0,0,0,0])
            self.assertEqual((r,d,t,i),(s,e,u,j))
            if d:break
        self.assertTrue(d)
        for rep in ['relative','cnn']:
            m,env=make(rep);m.learn(8)
            with tempfile.TemporaryDirectory() as root:
                path=Path(root)/'model';m.save(path)
                restored=type(m).load(path,env=env);restored.learn(8,reset_num_timesteps=False)
                self.assertEqual(restored.num_timesteps,16)
                self.assertEqual(restored.policy.action_net.in_features,146)

if __name__=='__main__':unittest.main()

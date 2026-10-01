import tempfile,pathlib,unittest
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.joint_movement import JointMovementDistribution,JointMovementPolicy
from touhou_ai.autumn_ablation import NumericalAblation,SpeedPolicy
class Tests(unittest.TestCase):
 def test_mapping_probabilities_and_gradient(self):
  logits=torch.zeros((18,18),requires_grad=True);d=JointMovementDistribution(logits)
  a=d.decode(torch.arange(18));self.assertEqual(len(set(map(tuple,a.tolist()))),18)
  self.assertTrue(torch.all(a[:,1]==0));self.assertTrue(torch.all(a[:,3]==0))
  torch.testing.assert_close(d.log_prob(a),torch.full((18,),-np.log(18),dtype=torch.float32))
  torch.testing.assert_close(d.entropy(),torch.full((18,),np.log(18),dtype=torch.float32))
  (-d.log_prob(a).mean()).backward();self.assertGreater(float(logits.grad.abs().sum()),0)
 def test_correlation_and_checkpoint(self):
  logits=torch.full((1,18),-1e9);logits[0,6]=0;logits[0,15]=0
  d=JointMovementDistribution(logits)
  sampled=set(tuple(d.sample()[0].tolist()) for _ in range(100))
  self.assertEqual(sampled,{(3,0,0,0),(7,0,1,0)})
  torch.set_num_threads(1);env=NumericalAblation(relative=True)
  m=PPO(JointMovementPolicy,env,seed=7,n_steps=8,batch_size=8,n_epochs=1,policy_kwargs={'share_features_extractor':False,'net_arch':[8]},device='cpu')
  shared={k:v.clone() for k,v in m.policy.state_dict().items() if not k.startswith('action_net')}
  base=PPO(SpeedPolicy,NumericalAblation(relative=True),seed=7,n_steps=8,batch_size=8,policy_kwargs={'share_features_extractor':False,'net_arch':[8]},device='cpu')
  for k,v in shared.items():torch.testing.assert_close(v,base.policy.state_dict()[k],rtol=0,atol=0)
  before=m.policy.action_net.weight.detach().clone();m.learn(8)
  self.assertFalse(torch.equal(before,m.policy.action_net.weight))
  obs,_=env.reset(seed=123);x,_=m.predict(obs,deterministic=True)
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'joint';m.save(p);loaded=PPO.load(p,device='cpu');y,_=loaded.predict(obs,deterministic=True)
   np.testing.assert_array_equal(x,y)
if __name__=='__main__':unittest.main()

import unittest
import numpy as np,torch
from touhou_ai.two_frame_plan_sim import PlanEnv,PlanDistribution,FrameVibration
from scripts.live_jitter_trial import Env

class PlanTests(unittest.TestCase):
    def test_held_physics_matches_existing(self):
        for seed in (7,17):
            a=PlanEnv();b=Env();a.reset(seed=seed);b.reset(seed=seed);rng=np.random.default_rng(seed)
            for _ in range(300):
                d=int(rng.integers(9));f=int(rng.integers(2))
                _,_,done,_,info=a.step([d,d,f]);_,_,old_done,_,old=b.step([d,0,f,0])
                np.testing.assert_array_equal(a.pos,b.pos);np.testing.assert_array_equal(a.xy,b.xy)
                self.assertEqual(a.frame,b.frame);self.assertEqual(done,old_done)
                if done:break

    def test_split_displacement_and_one_observation(self):
        e=PlanEnv(split=True);e.reset(seed=1);e.alive[:]=False
        calls=[];original=e.observe
        def observe():calls.append(1);return original()
        e.observe=observe;p=e.pos.copy();_,_,_,_,info=e.step([3,2,1])
        np.testing.assert_allclose(e.pos-p,[2+2/np.sqrt(2),-2/np.sqrt(2)])
        self.assertEqual(len(info['deltas']),2);self.assertEqual(len(calls),1)

    def test_first_frame_death_prevents_second_action(self):
        e=PlanEnv(split=True);e.reset(seed=1);e.xy[0]=e.pos;e.birth[0]=-15
        _,_,done,_,info=e.step([0,3,0]);self.assertTrue(done);self.assertEqual(info['frames'],1);self.assertEqual(len(info['deltas']),1)

    def test_joint_logprob_entropy_and_gradients(self):
        logits=torch.randn(4,92,requires_grad=True);d=PlanDistribution(logits,True)
        a=d.get_actions();lp=d.log_prob(a)
        joint=d.first.probs[:,:,None]*d.second.probs
        expected=joint[torch.arange(4),a[:,0],a[:,1]].log()+d.focus.log_prob(a[:,2])
        torch.testing.assert_close(lp,expected)
        torch.testing.assert_close(d.entropy(),torch.distributions.Categorical(probs=joint.flatten(1)).entropy()+d.focus.entropy())
        lp.sum().backward();g=logits.grad[:,9:90].reshape(4,9,9)
        for i in range(4):
            for j in range(9):
                if j!=a[i,0]:self.assertEqual(float(g[i,j].abs().sum()),0)
        h=PlanDistribution(logits.detach(),False);actions=h.get_actions();self.assertTrue(torch.equal(actions[:,0],actions[:,1]))
        torch.testing.assert_close(h.log_prob(actions),h.first.log_prob(actions[:,0])+h.focus.log_prob(actions[:,2]))

    def test_opposite_subframes_not_invisible(self):
        d=FrameVibration();penalty=sum(d.add([2 if i%2 else -2,0]) for i in range(120))
        self.assertGreater(penalty,0);self.assertLessEqual(penalty,.2+1e-6)
        self.assertEqual(d.add([0,0]),0)
        d.add([2,0]);self.assertEqual(d.add([2,0]),0)

if __name__=='__main__':unittest.main()

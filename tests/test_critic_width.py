import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from touhou_ai.autumn_training import build_model
from touhou_ai.critic_width import build
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from scripts.audit_cnn_layers import outputs

class CriticWidthTest(unittest.TestCase):
    def test_reduction_preserves_actor_rng_and_reload(self):
        from touhou_ai.critic_width import build64
        torch.set_num_threads(1)
        a,cls,_=build_model('cnn',7);rng=torch.get_rng_state().clone()
        b,_,_=build64(7)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in a.policy.parameters())-sum(p.numel() for p in b.policy.parameters()),16512)
        self.assertTrue(torch.equal(a.policy.mlp_extractor.value_net[2].weight[:64],b.policy.mlp_extractor.value_net[2].weight))
        self.assertAlmostEqual(float(a.policy.value_net.weight.norm()),float(b.policy.value_net.weight.norm()),places=6)
        obs=cls().reset(seed=5000)[0];batch={k:np.stack([v]) for k,v in obs.items()}
        p,_=outputs(a.policy,batch);q,w=outputs(b.policy,batch)
        np.testing.assert_array_equal(p,q)
        self.assertEqual({id(p) for g in b.policy.optimizer.param_groups for p in g['params']},{id(p) for p in b.policy.parameters()})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.zip';b.save(path)
            c=SeparateClipPPO.load(path,device='cpu')
            self.assertEqual(c.policy.mlp_extractor.latent_dim_vf,64)
            r,z=outputs(c.policy,batch);np.testing.assert_array_equal(q,r);np.testing.assert_array_equal(w,z)

    def test_matched_start_extra_capacity_and_reload(self):
        torch.set_num_threads(1)
        a,cls,_=build_model('cnn',7);rng=torch.get_rng_state().clone()
        b,_,_=build(7)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in b.policy.parameters())-sum(p.numel() for p in a.policy.parameters()),33024)
        obs=cls().reset(seed=5000)[0];batch={k:np.stack([v]) for k,v in obs.items()}
        p,v=outputs(a.policy,batch);q,w=outputs(b.policy,batch)
        np.testing.assert_array_equal(p,q);np.testing.assert_allclose(v,w,atol=1e-6,rtol=0)
        b.policy.predict_values(b.policy.obs_to_tensor(obs)[0]).sum().backward()
        self.assertGreater(float(b.policy.value_net.weight.grad[:,128:].abs().sum()),0)
        self.assertEqual({id(p) for g in b.policy.optimizer.param_groups for p in g['params']},{id(p) for p in b.policy.parameters()})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.zip';b.save(path)
            c=SeparateClipPPO.load(path,device='cpu')
            self.assertEqual(c.policy.mlp_extractor.latent_dim_vf,256)
            r,z=outputs(c.policy,batch);np.testing.assert_array_equal(q,r);np.testing.assert_array_equal(w,z)

if __name__=='__main__':unittest.main()

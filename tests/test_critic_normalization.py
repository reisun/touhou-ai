import tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from touhou_ai.critic_normalization import build
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from scripts.audit_cnn_layers import outputs

class CriticNormalizationTest(unittest.TestCase):
    def test_parameters_actor_rng_and_reload(self):
        torch.set_num_threads(1)
        control, cls, _ = build('control', 7)
        rng = torch.get_rng_state().clone()
        obs = cls().reset(seed=5000)[0]
        batch = {k: np.stack([v]) for k,v in obs.items()}
        p,_ = outputs(control.policy, batch)
        for variant in ('global','pre_tanh'):
            model,_,_ = build(variant,7)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(set(control.policy.state_dict()),set(model.policy.state_dict()))
            for key,value in control.policy.state_dict().items():
                self.assertTrue(torch.equal(value,model.policy.state_dict()[key]),key)
            q,v = outputs(model.policy,batch)
            np.testing.assert_array_equal(p,q)
            self.assertEqual({id(p) for g in model.policy.optimizer.param_groups for p in g['params']},
                             {id(p) for p in model.policy.parameters()})
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'model.zip';model.save(path)
                loaded=SeparateClipPPO.load(path,device='cpu')
                self.assertEqual(loaded.policy.norm_variant,variant)
                r,w=outputs(loaded.policy,batch)
                np.testing.assert_array_equal(q,r);np.testing.assert_array_equal(v,w)

if __name__=='__main__': unittest.main()

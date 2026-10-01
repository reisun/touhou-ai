import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from touhou_ai.autumn_training import build_reference_model as build_model
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from scripts.compare_critic_tanh import build
from scripts.audit_cnn_layers import outputs

class CriticNoFinalTanhTest(unittest.TestCase):
    def test_only_parameter_free_critic_activation_changes_and_survives_reload(self):
        torch.set_num_threads(1)
        baseline,env,_=build_model('cnn',7);variant=build(7)
        self.assertEqual(set(baseline.policy.state_dict()),set(variant.policy.state_dict()))
        for key,value in baseline.policy.state_dict().items():
            self.assertTrue(torch.equal(value,variant.policy.state_dict()[key]),key)
        left=dict(baseline.policy.named_modules());right=dict(variant.policy.named_modules())
        changes=[key for key in left if type(left[key]) is not type(right[key])]
        self.assertEqual(changes,['','mlp_extractor.value_net.3'])
        observation=env().reset(seed=7000)[0]
        obs={k:np.stack([v]) for k,v in observation.items()}
        p,_=outputs(baseline.policy,obs);q,v=outputs(variant.policy,obs)
        np.testing.assert_array_equal(p,q)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model';variant.save(path)
            loaded=SeparateClipPPO.load(path,device='cpu')
            self.assertIsInstance(loaded.policy.mlp_extractor.value_net[-1],torch.nn.Identity)
            r,w=outputs(loaded.policy,obs)
            np.testing.assert_array_equal(q,r);np.testing.assert_array_equal(v,w)

if __name__=='__main__':unittest.main()

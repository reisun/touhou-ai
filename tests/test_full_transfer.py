import tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from touhou_ai.live_action_grid import build_live_action_model
from touhou_ai.full_transfer import transfer_evasion
from touhou_ai.evasion_only import VERSION, WEIGHTS

class FullTransferTests(unittest.TestCase):
    def test_preserves_movement_enables_controls_and_scales_value(self):
        torch.set_num_threads(1)
        settings=dict(seed=7,n_steps=32,batch_size=32,n_epochs=1)
        old=build_live_action_model(settings)
        new=build_live_action_model(settings,evasion_only=False)
        obs={k:np.zeros((3,*space.shape),dtype=np.float32)
             for k,space in old.observation_space.spaces.items()}
        obs['bomb_clock'][1,0]=1/6
        tensor=old.policy.obs_to_tensor(obs)[0]
        manifest=dict(reward_version=VERSION,reward_weights=WEIGHTS,evasion_only=True,
                      cnn_architecture='narrow_action_grid',algorithm='SeparateClipPPO')
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'old.zip';old.save(checkpoint)
            transfer_evasion(checkpoint,manifest,new)
            with torch.no_grad():
                _,ov,_,od=old.policy.forward_with_distribution(tensor)
                _,nv,_,nd=new.policy.forward_with_distribution(tensor)
            for index in (0,2):torch.testing.assert_close(od[index].probs,nd[index].probs,rtol=0,atol=0)
            torch.testing.assert_close(nv,ov/60,rtol=1e-5,atol=1e-7)
            self.assertTrue((nd[1].probs[:,1]>0).all())
            self.assertTrue((nd[3].probs[:,1]>0).all())
            self.assertEqual(len(new.policy.optimizer.state),0)
            actions,_,logp=new.policy(tensor)
            _,evaluated,_=new.policy.evaluate_actions(tensor,actions)
            torch.testing.assert_close(logp,evaluated)
            self.assertEqual(actions[1,3],0)
            new.save(Path(d)/'new.zip')
            loaded=type(new).load(Path(d)/'new.zip')
            self.assertEqual(type(loaded.policy),type(new.policy))
            with self.assertRaises(ValueError):transfer_evasion(checkpoint,manifest|{'evasion_only':False},new)

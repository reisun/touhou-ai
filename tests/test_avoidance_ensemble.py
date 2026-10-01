import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
import numpy as np
from touhou_ai.avoidance_ensemble import Ensemble
class Tests(unittest.TestCase):
 def test_joint_choice_preserves_move_focus_pair(self):
  def ds(policy,obs):
   move=torch.tensor([[.9,.1,0,0,0,0,0,0,0]]) if policy==0 else torch.tensor([[.1,.9,0,0,0,0,0,0,0]])
   focus=torch.tensor([[.1,.9]]) if policy==0 else torch.tensor([[.9,.1]])
   return [SimpleNamespace(probs=move),None,SimpleNamespace(probs=focus),None]
  with patch('touhou_ai.avoidance_ensemble.distributions',side_effect=ds):
   a=Ensemble([SimpleNamespace(policy=0),SimpleNamespace(policy=1)])({})[0]
   self.assertIn((a[0],a[2]),[(0,1),(1,0)])
   np.testing.assert_array_equal(Ensemble([SimpleNamespace(policy=0)])({})[0],[0,0,1,0])
if __name__=='__main__':unittest.main()

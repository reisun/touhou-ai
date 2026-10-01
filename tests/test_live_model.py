import unittest
from touhou_ai.live_model import model_config,algorithm_class
class Tests(unittest.TestCase):
 def test_legacy_defaults_and_resume(self):
  old={'contract':'th10-dual-grid-v6'}
  self.assertEqual(model_config({},True,True,old),{'algorithm':'PPO','cnn_architecture':'dual_grid'})
 def test_new_config_rejects_legacy_and_preserves_own_resume(self):
  p={'evasion_policy_overrides':{'share_features_extractor':False,'algorithm':'SeparateClipPPO','cnn_architecture':'narrow_grid'}}
  c=model_config(p,True,True)
  with self.assertRaisesRegex(ValueError,'fresh campaign'):model_config(p,True,True,{'contract':'th10-dual-grid-v6'})
  self.assertEqual(model_config(p,True,True,{'contract':'th10-dual-grid-v6',**c}),c)
  self.assertEqual(algorithm_class(c['algorithm']).__name__,'SeparateClipPPO')
 def test_invalid_combinations(self):
  with self.assertRaises(ValueError):algorithm_class('unknown')
  with self.assertRaises(ValueError):model_config({'evasion_policy_overrides':{'algorithm':'SeparateClipPPO'}},True,True)
  with self.assertRaises(ValueError):model_config({'evasion_policy_overrides':{'cnn_architecture':'narrow_grid'}},True,False)
if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from scripts.analyze_critic_stability import bootstrap,describe

class StabilityAnalysisTest(unittest.TestCase):
    def test_paired_identical_results_have_no_difference(self):
        a=np.array([[0,0,1,0,1,0],[1,1,1,0,0,1],[1,0,0,0,0,0],[1,1,1,1,0,1]],float)
        result=bootstrap(a,a,draws=1000)
        np.testing.assert_array_equal(result['sd_ratio_95'],[1,1])
        np.testing.assert_array_equal(result['mean_difference_95'],[0,0])
    def test_lower_tail_and_sample_sd(self):
        result=describe(np.array([.05,.10,.15]))
        self.assertEqual(result['below10'],1)
        self.assertAlmostEqual(result['sd'],.05)

if __name__=='__main__':unittest.main()

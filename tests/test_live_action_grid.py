import unittest,tempfile,hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from touhou_ai.live_action_grid import LiveActionGridContract,live_risk_grid,CONTRACT,build_live_action_model,SPEC
from touhou_ai.spatial_input_candidates import CandidateEnv,risk_grid
from touhou_ai.live_model import model_config

class LiveActionGridTests(unittest.TestCase):
    def raw(self):
        e=CandidateEnv('action_grid');e.reset(seed=7);r=e.raw_observation();r['bullets']=[];return r
    def test_bullet_parity_and_previous_reward(self):
        from touhou_ai.dual_grid import DualGridContract
        e=CandidateEnv('action_grid');e.reset(seed=7)
        for t in range(80):
            raw=e.raw_observation();a=LiveActionGridContract().encode(raw,{'hit':-60});b=DualGridContract().encode(raw,{'hit':-60})
            for key in b:np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(a['action_grid'],risk_grid(raw))
            _,_,done,_,_=e.step([t%9,0,t%2,0])
            if done:break
    def test_static_lasers_swept_speed_and_activation(self):
        raw=self.raw();c={'origin':[8,320],'angle':np.pi/2,'length':20,'width':2,'active':True,'field_validated':False}
        raw['lasers']=[{'collision':c}]
        grid=LiveActionGridContract().encode(raw)['action_grid']
        self.assertEqual(grid[0,1,2],1);self.assertEqual(grid[1,1,2],0)
        self.assertEqual(grid[2,1,2],1);self.assertEqual(grid[0,1,0],0)
        c['origin']=[6,328];c['angle']=np.pi/4;c['length']=10;c['width']=1
        grid=live_risk_grid(raw);self.assertEqual(grid[0,1,2],1);self.assertEqual(grid[1,1,2],0)
        c['active']=False;self.assertFalse(live_risk_grid(raw).any())
        raw['lasers']=[{'collision':None}]
        with self.assertRaises(ValueError):LiveActionGridContract().encode(raw)
    def test_contract_selection_and_resume_integrity(self):
        from touhou_ai.live_learning import resume_manifest
        p={'evasion_policy_overrides':{'algorithm':'SeparateClipPPO','cnn_architecture':'narrow_action_grid','share_features_extractor':False}}
        selected=model_config(p,True,True)
        self.assertEqual(model_config(p,True,True,selected),selected)
        with self.assertRaisesRegex(ValueError,'fresh campaign'):model_config(p,True,True,selected|{'cnn_architecture':'narrow_grid'})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'artifacts/run';folder.mkdir(parents=True);model=folder/'model.zip';model.write_bytes(b'test')
            (folder/'status.json').write_text(json.dumps({'backend':'real_th10','contract':CONTRACT,'episodes':[{'checkpoint':'model.zip','reload_verified':True,'checkpoint_sha256':hashlib.sha256(b'test').hexdigest()}]}))
            with patch('touhou_ai.live_learning.ROOT',root):self.assertEqual(resume_manifest(model,dual_grid=True)[0]['contract'],CONTRACT)
    def test_initial_policy_matches_tested_offline_candidate(self):
        from scripts.compare_spatial_inputs import build
        from touhou_ai.autumn_training import REFERENCE_CONFIG
        from touhou_ai.model_monitor import model_metadata
        torch.set_num_threads(1);settings=json.loads(REFERENCE_CONFIG.read_text())['ppo']|{'seed':7}
        live=build_live_action_model(settings);offline,_=build('action_grid',7)
        for k,v in live.policy.state_dict().items():torch.testing.assert_close(v,offline.policy.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(model_metadata(live,CONTRACT,None)['bullet_scope'],SPEC)

if __name__=='__main__':unittest.main()

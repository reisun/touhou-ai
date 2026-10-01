import unittest,tempfile
from pathlib import Path
import numpy as np,torch
from touhou_ai.spatial_input_candidates import CandidateEnv,CandidateEncoder,risk_grid,paint,TILES,VARIANTS
from touhou_ai.autumn_ablation import GridAblation
from scripts.compare_spatial_inputs import build
from scripts.audit_cnn_layers import outputs
from touhou_ai.separate_clip_ppo import SeparateClipPPO

def scene(p=(8,330),v=(0,0)):
    e=CandidateEnv('action_grid');e.reset(seed=7);raw=e.raw_observation()
    raw['bullets']=[dict(position=list(p),velocity_raw=list(v),hitbox_raw=[4,4],flags_raw=2)]
    return raw

class SpatialInputTest(unittest.TestCase):
    def test_risk_speed_horizons_and_swept_collision(self):
        r=risk_grid(scene());self.assertEqual(r[0,1,2],1);self.assertEqual(r[1,1,2],0)
        r=risk_grid(scene((20,330),(-1,0)))
        self.assertEqual(r[0,1,2],0);self.assertEqual(r[2,1,2],1)
        r=risk_grid(scene((-5,330),(10,0)))
        self.assertEqual(r[0,1,1],1) # crossing between endpoints, stationary player
    def test_flags_and_wall_clipping(self):
        raw=scene();raw['bullets'][0]['flags_raw']=0
        self.assertFalse(risk_grid(raw).any())
        raw=scene((188,330));raw['player']['position']=[184,330]
        self.assertEqual(risk_grid(raw)[0,1,2],0)
    def test_one_pixel_preserves_subcell_structure(self):
        a=np.zeros((4,4),np.float32);b=np.zeros((2,2),np.float32)
        p=np.array([[1.,1.]]);half=np.array([[.5,.5]])
        paint(a,p,half,np.zeros(2),1);paint(b,p,half,np.zeros(2),2)
        self.assertAlmostEqual(float(a.sum()),1);self.assertAlmostEqual(float(b.sum()),.25)
        np.testing.assert_allclose(a.reshape(2,2,2,2).mean(axis=(1,3)),b)
    def test_geometry_uses_expanded_bullets_and_center_markers(self):
        raw=scene((6,330));o=CandidateEncoder('geometry').encode(raw)
        self.assertEqual(o['local_grid'].shape,(13,96,96))
        self.assertGreater(o['local_grid'][6].sum(),o['local_grid'][1].sum())
        self.assertGreater(o['local_grid'][9].sum(),0);self.assertGreater(o['local_grid'][11].sum(),0)
    def test_physics_and_raw_parity(self):
        for variant in VARIANTS:
            a=GridAblation();b=CandidateEnv(variant);a.reset(seed=43);b.reset(seed=43)
            for t in range(40):
                action=[t%9,0,t%2,0];oa,ra,da,_,ia=a.step(action);ob,rb,db,_,ib=b.step(action)
                self.assertEqual((ra,da,ia),(rb,db,ib));np.testing.assert_array_equal(a.pos,b.pos);np.testing.assert_array_equal(a.xy,b.xy)
                original=b.encoder.base.encode(b.raw_observation())
                for k in oa:np.testing.assert_array_equal(oa[k],original[k])
                self.assertTrue(b.observation_space.contains(ob))
                if da:break
    def test_neutral_added_inputs_and_roundtrip(self):
        torch.set_num_threads(1);base,cls=build('control',7);obs=cls().reset(seed=43)[0]
        p,v=outputs(base.policy,{k:x[None] for k,x in obs.items()})
        for variant in VARIANTS:
            m,env=build(variant,7);o=env().reset(seed=43)[0];batch={k:x[None] for k,x in o.items()}
            q,w=outputs(m.policy,batch)
            if variant!='pixel1':
                np.testing.assert_allclose(p,q,atol=1e-7);np.testing.assert_allclose(v,w,atol=1e-6)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'model';m.save(path);loaded=SeparateClipPPO.load(path,device='cpu')
                qq,ww=outputs(loaded.policy,batch);np.testing.assert_array_equal(q,qq);np.testing.assert_array_equal(w,ww)

if __name__=='__main__':unittest.main()

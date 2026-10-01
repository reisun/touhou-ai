"""Single-change CNN comparison, fresh training, existing paired controls."""
import sys,pathlib,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3.common.callbacks import BaseCallback
from touhou_ai.autumn_training import build_reference_model as build_model,REFERENCE_CONFIG as CONFIG
from touhou_ai.critic_no_final_tanh import CriticNoFinalTanhPolicy
from touhou_ai.autumn_ablation import GridAblation
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.simulation_speed import tune_cpu
from scripts.assess_autumn_holdout import assess_holdout
from scripts.audit_cnn_layers import inspect,outputs

def build(seed):
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    return SeparateClipPPO(CriticNoFinalTanhPolicy,GridAblation(),seed=seed,device='cpu',
        rollout_buffer_class=GridRolloutBuffer,verbose=0,
        policy_kwargs=dict(share_features_extractor=False,net_arch=dict(pi=[256,128],vf=[256,128]),features_extractor_class=NarrowGridFeatures),**config['ppo'])

class Record(BaseCallback):
    def __init__(self,path):super().__init__();self.path=path;self.rows=[]
    def _on_step(self):return True
    def _on_rollout_end(self):
        b=self.model.rollout_buffer;v=b.values.ravel();r=b.returns.ravel()
        self.rows.append(dict(steps=self.num_timesteps,value_std=float(v.std()),target_std=float(r.std()),
            explained_variance=float(1-np.var(r-v)/(np.var(r)+1e-12)),rmse=float(np.sqrt(np.mean((r-v)**2)))))
        self.path.write_text(json.dumps(self.rows,indent=2),encoding='utf-8')
        if self.num_timesteps%4096==0:print('training',self.num_timesteps,flush=True)

def main():
    root=pathlib.Path('artifacts/critic-no-final-tanh-20260929');root.mkdir(exist_ok=True)
    (root/'manifest.json').write_text(json.dumps(dict(change='critic final hidden Tanh -> Identity only',seeds=[7,17,27],steps=16384,representation='narrow CNN only',control='artifacts/autumn-separate-clip-20260928/cnn-switch-{seed}.zip',config=json.loads(CONFIG.read_text(encoding='utf-8')),evaluation='96 paired environment seeds5000..5095 and RNG19000..19095; previously used for model diagnostics, not training'),indent=2),encoding='utf-8')
    for seed in (7,17,27):
        dest=root/str(seed);dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        torch.set_num_threads(1);m=build(seed)
        baseline,_,_=build_model('cnn',seed)
        assert all(torch.equal(v,baseline.policy.state_dict()[k]) for k,v in m.policy.state_dict().items())
        # Restore training RNG after constructing the paired initial model.
        # Both constructions initialize identical parameter tensors and consume identical random draws.
        m=build(seed);tune_cpu(m);start=time.perf_counter()
        if (dest/'model.zip').exists():
            m=SeparateClipPPO.load(dest/'model.zip',device='cpu');torch.set_num_threads(1)
        else:
            m.learn(16384,callback=Record(dest/'learning.json'));m.save(dest/'model')
        loaded=SeparateClipPPO.load(dest/'model.zip',device='cpu')
        assert isinstance(loaded.policy.mlp_extractor.value_net[-1],torch.nn.Identity)
        assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in m.policy.state_dict().items())
        obs=dict(np.load(f'artifacts/cnn-layers-20260929/observations-{seed}.npz'))
        control=SeparateClipPPO.load(f'artifacts/autumn-separate-clip-20260928/cnn-switch-{seed}.zip',device='cpu')
        diagnosis=dict(control=inspect(control.policy,obs),variant=inspect(m.policy,obs))
        (dest/'layers.json').write_text(json.dumps(diagnosis,indent=2),encoding='utf-8')
        evaluations={}
        for mode in ('sample','greedy','frozen_sample','frozen_greedy'):
            evaluations[mode]=assess_holdout(m,GridAblation,mode=mode)
            (dest/'evaluation-progress.json').write_text(json.dumps(evaluations,indent=2),encoding='utf-8')
            print(seed,mode,evaluations[mode]['survival'],flush=True)
        row=dict(seed=seed,steps=m.num_timesteps,seconds=time.perf_counter()-start,evaluations=evaluations,
            save_reload_equal=True,initial_weights_equal=True)
        (dest/'result.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        print('COMPLETE',seed,flush=True)
if __name__=='__main__':main()

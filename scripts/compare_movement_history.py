"""Paired 2x2 history/penalty test from the same saved jittering live policy."""
import sys,pathlib,json,time,functools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.movement_history_sim import HistoryEnv,HistoryFeatures
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu
from scripts.compare_critic_tanh import Record
from scripts.sweep_jitter_penalty import summarize
from scripts import live_jitter_trial as trial

OUT=pathlib.Path('artifacts/movement-history-20261001')

def build(seed,penalty,history):
    source=json.loads((trial.OUT/'manifest.json').read_text())['checkpoint']
    old=SeparateClipPPO.load(source,device='cpu')
    settings=json.loads(pathlib.Path('configs/autumn-learning-baseline.json').read_text())['ppo']
    model=SeparateClipPPO(EvasionPolicy,HistoryEnv(penalty,history),seed=seed,device='cpu',
        rollout_buffer_class=GridRolloutBuffer,verbose=0,
        policy_kwargs=dict(share_features_extractor=False,net_arch=dict(pi=[256,128],vf=[256,128]),
                           features_extractor_class=HistoryFeatures),**settings)
    with torch.no_grad():
        for key,target in model.policy.state_dict().items():
            source=old.policy.state_dict()[key]
            if target.shape==source.shape:target.copy_(source)
            else:
                assert key.endswith('merge.0.weight') and target.shape[1]==source.shape[1]+24
                target.zero_();target[:,:source.shape[1]].copy_(source)
        obs,_=model.get_env().envs[0].unwrapped.reset(seed=5000)
        # Nonempty history must not perturb the matched initial policy/value.
        obs['movement_history'][:]=.5
        t,_=model.policy.obs_to_tensor(obs)
        oldt,_=old.policy.obs_to_tensor({k:v for k,v in obs.items() if k!='movement_history'})
        a=model.policy.get_distribution(t);b=old.policy.get_distribution(oldt)
        for i in (0,2):torch.testing.assert_close(a.distribution[i].probs,b.distribution[i].probs,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(model.policy.predict_values(t),old.policy.predict_values(oldt),atol=1e-5,rtol=1e-5)
    return model

def run(seed,history,penalty):
    dest=OUT/f'seed-{seed}-history-{int(history)}-penalty-{penalty}'
    dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);start=time.perf_counter()
    if (dest/'model.zip').exists():
        model=SeparateClipPPO.load(dest/'model.zip',device='cpu')
        assert model.num_timesteps==8192
    else:
        model=build(seed,penalty,history);tune_cpu(model,2)
        model.learn(8192,callback=Record(dest/'learning.json'));model.save(dest/'model.zip')
    loaded=SeparateClipPPO.load(dest/'model.zip',device='cpu')
    assert all(torch.equal(v,loaded.policy.state_dict()[k]) and torch.isfinite(v).all() for k,v in model.policy.state_dict().items())
    trial.Env=functools.partial(HistoryEnv,penalty=penalty,history=history)
    result=dict(seed=seed,history=history,penalty=penalty,steps=model.num_timesteps,initial_policy_matched=True)
    for task,y in [('standard',None),('challenge',160)]:
        rows=trial.evaluate(model,n=64,start_y=y)
        (dest/f'{task}.json').write_text(json.dumps(rows))
        result[task]=summarize(rows)
    result['seconds']=time.perf_counter()-start
    (dest/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':run(int(sys.argv[1]),bool(int(sys.argv[2])),float(sys.argv[3]))

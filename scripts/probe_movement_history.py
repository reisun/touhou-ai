"""Counterfactual input sensitivity on replayed evaluation states (not new rollouts)."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.movement_history_sim import HistoryEnv
from touhou_ai.separate_clip_ppo import SeparateClipPPO
OUT=pathlib.Path('artifacts/movement-history-20261001')
torch.set_num_threads(1)
results=[]
for seed in (7,17,27):
    for penalty in (0.,.1):
        dest=OUT/f'seed-{seed}-history-1-penalty-{penalty}'
        if not (dest/'result.json').exists():continue
        model=SeparateClipPPO.load(dest/'model.zip',device='cpu')
        rows=[]
        for task,y in [('standard',330),('challenge',160)]:
            for episode in json.loads((dest/f'{task}.json').read_text())[:4]:
                env=HistoryEnv(penalty,True);env.reset(seed=episode['seed']);env.pos[1]=y;obs=env.observe()
                for t,step in enumerate(episode['trace'][:121]):
                    if t in (6,30,60,90,120):rows.append(obs)
                    obs,_,done,_,_=env.step(step['action'])
                    if done:break
        batch={k:np.stack([o[k] for o in rows]) for k in rows[0]}
        zero=dict(batch,movement_history=np.zeros_like(batch['movement_history']))
        movement_only_zero=np.zeros_like(batch['movement_history'])
        movement_only_zero[:,:,3]=batch['movement_history'][:,:,3]
        keep_valid=dict(batch,movement_history=movement_only_zero)
        with torch.inference_mode():
            a,_=model.policy.obs_to_tensor(batch);b,_=model.policy.obs_to_tensor(zero)
            c,_=model.policy.obs_to_tensor(keep_valid)
            p=model.policy.get_distribution(a).distribution;q=model.policy.get_distribution(b).distribution
            z=model.policy.get_distribution(c).distribution
            result=dict(seed=seed,penalty=penalty,observations=len(rows),
                direction_probability_tv_mean=float(((p[0].probs-q[0].probs).abs().sum(1)/2).mean()),
                direction_argmax_changed=float((p[0].probs.argmax(1)!=q[0].probs.argmax(1)).float().mean()),
                movement_only_tv_mean=float(((p[0].probs-z[0].probs).abs().sum(1)/2).mean()),
                movement_only_argmax_changed=float((p[0].probs.argmax(1)!=z[0].probs.argmax(1)).float().mean()),
                focus_probability_change_mean=float((p[2].probs[:,1]-q[2].probs[:,1]).abs().mean()),
                value_change_mean=float((model.policy.predict_values(a)-model.policy.predict_values(b)).abs().mean()))
        results.append(result)
(OUT/'history-sensitivity.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))

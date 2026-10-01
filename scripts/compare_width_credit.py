"""Fresh independent changes against verified v4 controls."""
import json,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from touhou_ai.simulation_speed import tune_cpu
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from scripts.compare_critic_tanh import Record
from scripts.assess_autumn_holdout import assess_holdout
from scripts.compare_critic_normalization import diagnosis

if __name__=='__main__':
    variant=sys.argv[1];seed=int(sys.argv[2]);assert seed in (7,17,27)
    if variant=='width':from touhou_ai.critic_width import build
    elif variant=='width64':from touhou_ai.critic_width import build64 as build
    elif variant=='credit':from touhou_ai.rollout_advantage import build
    elif variant=='adam05':from touhou_ai.adam_memory import build
    else:raise ValueError(variant)
    dest=pathlib.Path('artifacts/width-credit-20260930')/variant/str(seed)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():sys.exit(0)
    torch.set_num_threads(1);m,cls,config=build(seed)
    control=json.loads((pathlib.Path('artifacts/critic-normalization-20260930/control')/str(seed)/'result.json').read_text())
    assert control['config']==config and control['steps']==16384
    (dest/'manifest.json').write_text(json.dumps(dict(variant=variant,seed=seed,steps=16384,config=config,
        control=f'artifacts/critic-normalization-20260930/control/{seed}',
        optimizer_betas=[list(g['betas']) for g in m.policy.optimizer.param_groups],
        evaluation='96 existing diagnostic patterns; sample, greedy, frozen_greedy',
        note='single change; not combined; real game unchanged'),indent=2))
    tune_cpu(m,update_threads=4);start=time.perf_counter()
    print('START',variant,seed,flush=True)
    m.learn(16384,callback=Record(dest/'learning.json'));m.save(dest/'model.zip')
    loaded=type(m).load(dest/'model.zip',device='cpu')
    assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in m.policy.state_dict().items())
    assert [g['betas'] for g in loaded.policy.optimizer.param_groups]==[g['betas'] for g in m.policy.optimizer.param_groups]
    assert all(torch.isfinite(p).all() for p in m.policy.parameters())
    training_seconds=time.perf_counter()-start;evaluations={}
    for mode in ('sample','greedy','frozen_greedy'):
        evaluations[mode]=assess_holdout(m,cls,n=96,mode=mode)
        print('EVAL',variant,seed,mode,evaluations[mode]['survival'],flush=True)
    (dest/'result.json').write_text(json.dumps(dict(variant=variant,seed=seed,steps=m.num_timesteps,
        config=config,training_seconds=training_seconds,seconds=time.perf_counter()-start,
        evaluations=evaluations,diagnostic=diagnosis(m.policy,cls),save_reload_equal=True),indent=2))

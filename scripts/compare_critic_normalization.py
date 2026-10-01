"""Bounded paired fresh-training comparison, current v4 CNN only."""
import json,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from touhou_ai.critic_normalization import build, VARIANTS
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from scripts.compare_critic_tanh import Record
from scripts.assess_autumn_holdout import assess_holdout
from touhou_ai.simulation_speed import tune_cpu

OUT=pathlib.Path('artifacts/critic-normalization-20260930')

def diagnosis(policy, cls):
    observations=[]
    # Common observations from fixed actions, independent of trained policy.
    for seed in range(7100,7108):
        env=cls();obs,_=env.reset(seed=seed)
        for t in range(300):
            if t%30==0:observations.append(obs)
            obs,_,done,trunc,_=env.step([0,0,0,0])
            if done or trunc:break
    parts=[]
    layer=policy.mlp_extractor.value_net[-1]
    handle=layer.register_forward_hook(lambda m,a,y:parts.append(y.detach().numpy()))
    with torch.no_grad():
        for o in observations:
            policy.predict_values(policy.obs_to_tensor(o)[0])
    handle.remove();a=np.concatenate(parts)
    return dict(samples=len(a),last_tanh_saturation=float((abs(a)>.99).mean()),
                scene_std_rms=float(np.sqrt(np.mean(a.std(0)**2))))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(seeds=[7,17,27],steps=16384,
        variants=list(VARIANTS),normalization='LayerNorm, affine=False, critic only',update_threads=4,
        comparison='independent single changes; same initial parameters/RNG; env seed7',
        evaluation='96 existing diagnostic patterns, sample and greedy; not untouched test',
        limitation='v4 death-only simulator does not reproduce live full-reward distribution'),indent=2))
    seeds=(int(sys.argv[1]),) if len(sys.argv)>1 else (7,17,27)
    assert all(s in (7,17,27) for s in seeds)
    for seed in seeds:
        for variant in VARIANTS:
            dest=OUT/variant/str(seed);dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():continue
            torch.set_num_threads(1)
            model,cls,config=build(variant,seed)
            tune_cpu(model,update_threads=4)
            start=time.perf_counter();print('START',variant,seed,flush=True)
            model.learn(16384,callback=Record(dest/'learning.json'))
            model.save(dest/'model.zip')
            loaded=SeparateClipPPO.load(dest/'model.zip',device='cpu')
            assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
            assert all(torch.isfinite(p).all() for p in model.policy.parameters())
            training_seconds=time.perf_counter()-start
            evaluations={}
            for mode in ('sample','greedy'):
                evaluations[mode]=assess_holdout(model,cls,n=96,mode=mode)
                print('EVAL',variant,seed,mode,evaluations[mode]['survival'],flush=True)
            (dest/'result.json').write_text(json.dumps(dict(variant=variant,seed=seed,config=config,
                steps=model.num_timesteps,training_seconds=training_seconds,
                seconds=time.perf_counter()-start,evaluations=evaluations,
                diagnostic=diagnosis(model.policy,cls),save_reload_equal=True),indent=2))

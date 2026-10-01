"""Sequential experiments, each compared to the adopted v3, never cumulative."""
import sys,pathlib,json,time,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from touhou_ai.risk_refinements import build
from touhou_ai.simulation_speed import tune_cpu
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from scripts.compare_critic_tanh import Record
from scripts.assess_autumn_holdout import assess_holdout

OUT=pathlib.Path('artifacts/risk-refinements-20260929')
if __name__=='__main__':
    variant=sys.argv[1]
    assert variant in ('control','urgency','continuous','scale')
    seeds=(int(sys.argv[2]),) if len(sys.argv)>2 else (7,17,27)
    assert all(seed in (7,17,27) for seed in seeds)
    for seed in seeds:
        dest=OUT/variant/str(seed);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():continue
        torch.set_num_threads(1)
        m,cls,config=build(variant,seed)
        manifest=dict(variant=variant,seed=seed,steps=16384,config=config,
          change={'control':'adopted v3','urgency':'binary -> 1-contact_time/(horizon+1); same move2F-stop path',
                  'continuous':'binary collision; continue selected movement for16F, clip at walls',
                  'scale':'death -60 -> -1 only; other PPO coefficients unchanged'}[variant],
          physics_sha256=hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),
          evaluation='96 paired known diagnostic seeds5000..5095, four modes; not untouched test set',
          training_environment_initial_seed=7,
          control='fresh adopted v3; independent single changes; three initialization/action-RNG seeds; same environment initial seed7; no live modifications')
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
        start=time.perf_counter();tune_cpu(m,update_threads=4)
        print('START',variant,seed,flush=True)
        m.learn(16384,callback=Record(dest/'learning.json'));m.save(dest/'model')
        saved=SeparateClipPPO.load(dest/'model.zip',device='cpu')
        assert all(torch.equal(v,saved.policy.state_dict()[k]) for k,v in m.policy.state_dict().items())
        training_seconds=time.perf_counter()-start
        evaluations={}
        for mode in ('sample','greedy','frozen_sample','frozen_greedy'):
            evaluations[mode]=assess_holdout(m,cls,mode=mode)
            (dest/'evaluation-progress.json').write_text(json.dumps(evaluations))
            print('EVAL',variant,seed,mode,evaluations[mode]['survival'],flush=True)
        (dest/'result.json').write_text(json.dumps(dict(variant=variant,seed=seed,training_seconds=training_seconds,
            seconds=time.perf_counter()-start,evaluations=evaluations,save_reload_equal=True),indent=2))

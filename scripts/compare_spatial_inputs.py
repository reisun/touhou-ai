"""Four-arm, matched-initialization CNN input comparison; offline only."""
import sys,pathlib,json,time,functools,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.autumn_training import build_reference_model as build_model,REFERENCE_CONFIG as CONFIG
from touhou_ai.autumn_ablation import GridAblation,SpeedPolicy
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.spatial_input_candidates import CandidateEnv,CandidateFeatures,VARIANTS,build_candidate_model
from touhou_ai.simulation_speed import tune_cpu,distributions
from scripts.compare_critic_tanh import Record
from scripts.assess_autumn_holdout import assess_holdout

OUT=pathlib.Path('artifacts/spatial-inputs-20260929')
SEEDS=(7,17,27)
JOBS=[(v,s) for v in ('pixel1','control','action_grid','geometry') for s in SEEDS]

def build(variant,seed):
    baseline,_,config=build_model('cnn',seed)
    if variant=='control':return baseline,GridAblation
    return build_candidate_model(variant,seed,baseline,config)

def prepare():
    OUT.mkdir(exist_ok=True)
    manifest=dict(variants=['control',*VARIANTS],seeds=SEEDS,steps=16384,jobs=JOBS,
        config=json.loads(CONFIG.read_text(encoding='utf-8')),update_threads=4,inference_threads=1,
        evaluation='96 paired episodes, environment5000..5095 and action RNG19000..19095; sample,greedy,frozen_sample,frozen_greedy',
        primary='Stochastic survival and mean duration, paired across three seeds; frozen observation control, input throughput and memory secondary',
        stopping='Complete all 12 jobs; do not combine candidates or stop on favorable outcomes',
        initialization='Shared tensors copied from same-seed baseline; added channel/readout weights zero; new branch internal weights random; baseline post-initialization RNG restored',
        physics_sha256=hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),
        approximation='Only observed positions/velocities and hitboxes. No future emissions or turns. Action grid predicts bullets only; no lasers in this simulator task.',
        not_changed='live learner, standard configuration, action9+2, death-only reward, critic Tanh')
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==json.loads(json.dumps(manifest))
    else:path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def work(variant,seed,dest):
    torch.set_num_threads(1);print('START',variant,seed,flush=True);began=time.perf_counter()
    m,cls=build(variant,seed);tune_cpu(m,update_threads=4)
    m.learn(16384,callback=Record(dest/'learning.json'));m.save(dest/'model')
    train_seconds=time.perf_counter()-began
    saved=SeparateClipPPO.load(dest/'model.zip',device='cpu')
    assert all(torch.equal(v,saved.policy.state_dict()[k]) for k,v in m.policy.state_dict().items())
    evaluation={}
    for mode in ('sample','greedy','frozen_sample','frozen_greedy'):
        evaluation[mode]=assess_holdout(m,cls,mode=mode)
        (dest/'evaluation-progress.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
        print('EVAL',variant,seed,mode,evaluation[mode]['survival'],flush=True)
    row=dict(variant=variant,seed=seed,steps=m.num_timesteps,training_seconds=train_seconds,seconds=time.perf_counter()-began,
        parameters=sum(p.numel() for p in m.policy.parameters()),evaluations=evaluation,save_reload_equal=True)
    (dest/'result.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('DONE',variant,seed,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        for variant,seed in JOBS:
            dest=OUT/variant/str(seed);dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():continue
            try:(dest/'claim').mkdir()
            except FileExistsError:continue
            work(variant,seed,dest)

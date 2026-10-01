"""Prespecified CNN-only final-Tanh replication; independent 9+2 throughout."""
import sys,pathlib,json,time,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_training import REFERENCE_CONFIG as CONFIG
from touhou_ai.simulation_speed import tune_cpu
from scripts.compare_critic_tanh import build,Record
from scripts.replicate_cnn_output_stability import evaluate

OUT=pathlib.Path('artifacts/critic-stability-20260929')
FRESH=[37,47,57,67,77,87,97]
OLD=[7,17,27]

def prepare():
    OUT.mkdir(exist_ok=True)
    manifest=dict(fresh_seeds=FRESH,exploratory_seeds=OLD,steps=16384,evaluation_episodes=192,
        environment_seeds=[5000,5191],action_random_seeds=[19000,19191],
        change='Only final critic hidden Tanh -> Identity; CNN independent movement9 + focus2',
        config=json.loads(CONFIG.read_text(encoding='utf-8')),
        primary='Seven additional paired initializations, SD ratio of stochastic survival; mean and lower tail also reported',
        criteria=dict(stability='upper 95% paired seed+case bootstrap interval for SD ratio < 1',
            mean_noninferiority='lower 95% interval for mean difference > -0.02',
            adoption='both conditions, with lower tail inspected'),
        stopping='Complete all seven additional initializations, irrespective of interim results',
        reuse='Existing original-Tanh controls and their 192 evaluations; original three no-Tanh models need 96 additional evaluations',
        caveat='Evaluation cases used before, excluded from training. Additional initializations are new to the no-Tanh comparison, not all new baseline training.',
        physics_sha256=hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest())
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==manifest
    else:path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for seed in OLD+FRESH:
        src=pathlib.Path(f'artifacts/cnn-output-stability-20260929/{seed}/independent/result.json')
        r=json.loads(src.read_text(encoding='utf-8'));r['variant']='control'
        p=OUT/str(seed)/'control';p.mkdir(parents=True,exist_ok=True)
        (p/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')

def work(seed):
    p=OUT/str(seed)/'no_tanh';p.mkdir(parents=True,exist_ok=True)
    if (p/'result.json').exists():return
    torch.set_num_threads(1);start=time.perf_counter();print('START',seed,flush=True)
    if seed in OLD:
        modelpath=pathlib.Path(f'artifacts/critic-no-final-tanh-20260929/{seed}/model.zip')
        m=SeparateClipPPO.load(modelpath,device='cpu')
    else:
        modelpath=p/'model.zip'
        if modelpath.exists():m=SeparateClipPPO.load(modelpath,device='cpu')
        else:
            m=build(seed);tune_cpu(m)
            m.learn(16384,callback=Record(p/'learning.json'));m.save(modelpath)
    assert m.num_timesteps==16384
    assert isinstance(m.policy.mlp_extractor.value_net[-1],torch.nn.Identity)
    torch.set_num_threads(1);episodes=[]
    for offset in (0,96):
        panel=p/f'panel-{offset}.json'
        if panel.exists():rows=json.loads(panel.read_text(encoding='utf-8'))
        elif seed in OLD and offset==0:
            old=json.loads(pathlib.Path(f'artifacts/critic-no-final-tanh-20260929/{seed}/result.json').read_text(encoding='utf-8'))
            rows=[{k:e[k] for k in ('seed','success','frames')} for e in old['evaluations']['sample']['episodes']]
        else:rows=evaluate(m,'independent',start=offset,n=96)
        panel.write_text(json.dumps(rows),encoding='utf-8');episodes.extend(rows)
        print('EVAL',seed,offset,float(np.mean([r['success'] for r in rows])),flush=True)
    episodes=sorted(episodes,key=lambda r:r['seed'])
    assert [e['seed'] for e in episodes]==list(range(5000,5192))
    result=dict(seed=seed,variant='no_tanh',steps=m.num_timesteps,fresh=seed in FRESH,checkpoint=str(modelpath),
        episodes=episodes,survival=float(np.mean([e['success'] for e in episodes])),
        seconds_alive=float(np.mean([e['frames'] for e in episodes])/60),wall_seconds=time.perf_counter()-start)
    (p/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DONE',seed,result['survival'],flush=True)

if __name__=='__main__':
    command=sys.argv[1]
    if command=='prepare':prepare()
    else:
        seeds=OLD if command=='old' else FRESH[int(command)::2]
        for seed in seeds:work(seed)

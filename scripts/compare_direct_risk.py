"""Direct observable-risk input, with adopted baselines and learned independent actions."""
import sys,pathlib,json,time,functools,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3 import PPO
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.direct_risk import DirectRiskInput,DirectRiskPolicy,WithoutActionRisk,ACTION_RISK
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu,distributions
from scripts.assess_autumn_holdout import assess_holdout
from scripts.continue_dodge_comparison import Progress
from scripts.compare_risk_auxiliary import make_probe

def risk_probe(model,observations):
    rows={};model.policy.set_training_mode(False)
    labels=np.stack([o['risk_targets'] for o in observations]);permutation=np.random.default_rng(83).permutation(len(labels))
    for mode in ['normal','zero','shuffled']:
        mass=[];greedy=[]
        for start in range(0,len(observations),16):
            group=observations[start:start+16]
            batch={k:np.stack([o[k] for o in group]) for k in group[0] if k!='risk_targets'}
            y=labels[start:start+len(group)]
            batch[ACTION_RISK]=y if mode=='normal' else np.zeros_like(y) if mode=='zero' else labels[permutation[start:start+len(group)]]
            with torch.inference_mode():ds=distributions(model.policy,batch)
            p=(ds[0].probs[:,:,None]*ds[2].probs[:,None,:]).reshape(-1,18).numpy()
            mass.extend((p*y).sum(1).tolist());greedy.extend(y[np.arange(len(y)),p.argmax(1)].tolist())
        rows[mode]={'approx_danger_mass':float(np.mean(mass)),'greedy_approx_danger_fraction':float(np.mean(greedy))}
    return {'mixed_scenes':len(observations),'uniform_approx_danger_mass':float(labels.mean()),'modes':rows}

def main():
    rep=sys.argv[1]
    if rep not in ['relative','cnn']:raise ValueError(rep)
    out=pathlib.Path('artifacts/autumn-direct-risk-20260928')/rep;out.mkdir(parents=True,exist_ok=True)
    config=json.loads(pathlib.Path('configs/autumn-learning-reference-v2.json').read_text(encoding='utf-8'))
    separate=config['representations'][rep]['algorithm']=='SeparateClipPPO'
    baseline=json.loads(pathlib.Path('artifacts/autumn-separate-clip-20260928/holdout.json').read_text())
    baseline=[r for r in baseline if r['representation']==rep and r['variant']==('separate' if separate else 'baseline')]
    assert len(baseline)==3
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
    manifest={'representation':rep,'seeds':[7,17,27],'steps':16384,'baseline_config':config,'physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'change':'18 observable constant-velocity binary risk indicators concatenate to actor latent immediately before action_net; added weights initialize0; critic input unchanged; no auxiliary loss or action mask','action':'independent direction9 and focus2; shot/bomb disabled','risk':'same2F move+14F stationary predictor as prior auxiliary task; current observations only','evaluation':'96 environment seeds5000..5095; RNG19000..19095; main4 modes plus zero_risk_sample','probe':'same108 mixed scenes; risk input normal/zero/permuted with original observations unchanged'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
    env_cls=functools.partial(DirectRiskInput,cls)
    torch.set_num_threads(1);observations=make_probe(cls)
    algorithm=SeparateClipPPO if separate else PPO
    for seed in [7,17,27]:
        if any(r['seed']==seed for r in rows):continue
        print('START',rep,seed,flush=True);start=time.perf_counter()
        kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]},'features_extractor_class':WithoutActionRisk if rep=='relative' else NarrowGridFeatures}
        model=algorithm(DirectRiskPolicy,env_cls(),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0,**config['ppo'])
        tune_cpu(model);before=risk_probe(model,observations)
        model.learn(16384,callback=Progress());model.save(out/f'direct-{seed}')
        after=risk_probe(model,observations);print('PROBE',rep,seed,json.dumps(after),flush=True)
        evaluations={}
        for mode in ['sample','greedy','frozen_sample','frozen_greedy','zero_risk_sample']:
            factory=functools.partial(DirectRiskInput,cls,enabled=mode!='zero_risk_sample')
            evaluations[mode]=assess_holdout(model,factory,mode='sample' if mode=='zero_risk_sample' else mode)
            print('EVAL',rep,seed,mode,round(evaluations[mode]['survival']*100,2),round(evaluations[mode]['mean_frames']/60,3),flush=True)
        row={'representation':rep,'seed':seed,'probe_before':before,'probe_after':after,'evaluations':evaluations,'seconds':time.perf_counter()-start,'risk_weight_norm':float(model.policy.action_net.weight[:,-18:].detach().norm()),'training_metrics':{k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/')}}
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'complete','adopted':False}))
    print('COMPLETE',rep,flush=True)
if __name__=='__main__':main()

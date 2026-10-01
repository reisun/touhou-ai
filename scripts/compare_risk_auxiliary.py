"""Matched three-seed auxiliary-risk experiment, adopted per-representation baselines."""
import sys,pathlib,json,time,functools,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.risk_auxiliary import RiskTargets,RiskAuxPolicy,WithoutRiskTargets
from touhou_ai.risk_aux_ppo import RiskAuxPPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import tune_cpu
from scripts.assess_autumn_holdout import assess_holdout
from scripts.continue_dodge_comparison import Progress

def make_probe(cls):
    observations=[]
    for ep in range(12):
        e=RiskTargets(cls);o,_=e.reset(seed=7000+ep);rng=np.random.default_rng(23000+ep)
        for step in range(300):
            y=o['risk_targets']
            if step%2==0 and 0<y.sum()<18:observations.append(o)
            o,_,done,_,_=e.step([int(rng.integers(9)),0,int(rng.integers(2)),0])
            if done:break
    if len(observations)>256:
        idx=np.random.default_rng(31).choice(len(observations),256,replace=False)
        observations=[observations[i] for i in sorted(idx)]
    if not observations:raise AssertionError('No mixed-risk probe scenes')
    return observations

def probe(model,observations):
    stats=[];all_logits=[];all_targets=[]
    model.policy.set_training_mode(False)
    for start in range(0,len(observations),16):
        group=observations[start:start+16];batch={k:np.stack([o[k] for o in group]) for k in group[0]}
        tensor,_=model.policy.obs_to_tensor(batch)
        with torch.inference_mode():
            pi,_=model.policy._latents(tensor);logits=model.policy.risk_head(pi).numpy()
        y=batch['risk_targets'];positive=y.sum(1);negative=18-positive
        recall=((logits>=0)*y).sum(1)/positive;specificity=((logits<0)*(1-y)).sum(1)/negative
        bce=.5*((np.logaddexp(0,-logits)*y).sum(1)/positive+(np.logaddexp(0,logits)*(1-y)).sum(1)/negative)
        stats.extend(zip(recall,specificity,bce));all_logits.append(logits);all_targets.append(y)
    a=np.array(stats);z=np.concatenate(all_logits);y=np.concatenate(all_targets)
    pos=y.sum(0);neg=len(y)-pos;ok=(pos>0)&(neg>0)
    action_accuracy=.5*(((z>=0)*y).sum(0)/np.maximum(pos,1)+((z<0)*(1-y)).sum(0)/np.maximum(neg,1))
    return {'action_balanced_accuracy':float(action_accuracy[ok].mean()),'mixed_scenes':len(a),'balanced_accuracy':float(a[:,:2].mean()),'danger_recall':float(a[:,0].mean()),'safe_recall':float(a[:,1].mean()),'balanced_bce':float(a[:,2].mean())}

def main():
    rep=sys.argv[1]
    if rep not in ['relative','cnn']:raise ValueError(rep)
    balance=sys.argv[2] if len(sys.argv)>2 else 'scene'
    seeds=[7,17,27] if balance=='scene' else [7]
    out=pathlib.Path('artifacts/autumn-risk-aux-20260928' if balance=='scene' else 'artifacts/autumn-risk-aux-action-balanced-20260928')/rep;out.mkdir(parents=True,exist_ok=True)
    config=json.loads(pathlib.Path('configs/autumn-learning-reference-v2.json').read_text(encoding='utf-8'))
    separate=config['representations'][rep]['algorithm']=='SeparateClipPPO'
    baseline=json.loads(pathlib.Path('artifacts/autumn-separate-clip-20260928/holdout.json').read_text())
    baseline=[r for r in baseline if r['representation']==rep and r['variant']==('separate' if separate else 'baseline')]
    assert len(baseline)==3
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
    manifest={'representation':rep,'seeds':seeds,'risk_balance':balance,'steps':16384,'aux_coef':.1,'separate_clip':separate,'baseline_config':config,'physics_sha256':hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest(),'target':'constant observed velocity; action2F then stationary14F; AABB quarter-frame; selected scene/action balancing; labels excluded from actor/critic inputs','evaluation':'96 seeds5000..5095 and RNG19000..19095; sample,greedy,frozen_sample,frozen_greedy','probe':'mixed scenes from 12 random-action episodes seeds7000..7011; not fed into training; evaluates prediction of approximate target, not real future collision'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
    train_cls=functools.partial(RiskTargets,cls);eval_cls=functools.partial(RiskTargets,cls,labels=False)
    torch.set_num_threads(1);observations=make_probe(cls);print('PROBE',rep,len(observations),flush=True)
    for seed in seeds:
        if any(r['seed']==seed for r in rows):continue
        print('START',rep,seed,flush=True);start=time.perf_counter()
        kw={'risk_balance':balance,'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]},'features_extractor_class':WithoutRiskTargets if rep=='relative' else NarrowGridFeatures}
        model=RiskAuxPPO(RiskAuxPolicy,train_cls(),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,aux_coef=.1,separate_clip=separate,verbose=0,**config['ppo'])
        tune_cpu(model);before=probe(model,observations)
        model.learn(16384,callback=Progress());model.save(out/f'aux-{seed}')
        after=probe(model,observations);print('PROBE_RESULT',rep,seed,before,after,flush=True)
        evaluations={}
        for mode in ['sample','greedy','frozen_sample','frozen_greedy']:
            evaluations[mode]=assess_holdout(model,eval_cls,mode=mode)
            print('EVAL',rep,seed,mode,round(evaluations[mode]['survival']*100,2),round(evaluations[mode]['mean_frames']/60,3),flush=True)
        row={'representation':rep,'seed':seed,'probe_before':before,'probe_after':after,'evaluations':evaluations,'seconds':time.perf_counter()-start,'training_metrics':{k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/')}}
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'complete','adopted':False}))
    print('COMPLETE',rep,flush=True)
if __name__=='__main__':main()

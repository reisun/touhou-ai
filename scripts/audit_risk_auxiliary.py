"""Probe conditional perception and policy risk mass, using observable approximation only."""
import sys,pathlib,json,functools,argparse
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch,numpy as np
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import NumericalAblation,GridAblation
from touhou_ai.risk_auxiliary import RiskTargets
from touhou_ai.simulation_speed import distributions
from scripts.compare_risk_auxiliary import make_probe

def balanced(logits,y):
    return .5*(((logits>=0)*y).sum(1)/y.sum(1)+((logits<0)*(1-y)).sum(1)/(18-y.sum(1)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('representations',nargs='*');ap.add_argument('--balance',choices=['scene','action'],default='scene');args=ap.parse_args()
    torch.set_num_threads(1);out=pathlib.Path('artifacts/autumn-risk-aux-20260928' if args.balance=='scene' else 'artifacts/autumn-risk-aux-action-balanced-20260928');results=[]
    for rep in (args.representations or ['relative','cnn']):
        cls=functools.partial(NumericalAblation,relative=True) if rep=='relative' else GridAblation
        observations=make_probe(cls);initial,_=RiskTargets(cls,labels=False).reset(seed=7000)
        bases={r['seed']:r for r in json.loads((out/rep/'baseline.json').read_text())}
        for seed in [7,17,27]:
            aux=out/rep/f'aux-{seed}.zip'
            if not aux.exists():continue
            for variant,path in [('baseline',pathlib.Path(bases[seed]['checkpoint'])),('aux',aux)]:
                m=PPO.load(path,device='cpu');p=m.policy;p.set_training_mode(False)
                risk_mass=[];greedy=[];acc=[];frozen_acc=[];uniform=[];targets=[];predictions=[]
                initial_input={k:initial[k] for k in m.observation_space.spaces}
                t0,_=p.obs_to_tensor(initial_input)
                with torch.inference_mode():
                    if variant=='aux':l0=p.risk_head(p._latents(t0)[0]).numpy()
                for start in range(0,len(observations),16):
                    group=observations[start:start+16];y=np.stack([o['risk_targets'] for o in group])
                    batch={k:np.stack([o[k] for o in group]) for k in m.observation_space.spaces}
                    tensor,_=p.obs_to_tensor(batch)
                    with torch.inference_mode():
                        ds=distributions(p,batch)
                        probs=(ds[0].probs[:,:,None]*ds[2].probs[:,None,:]).reshape(-1,18).numpy()
                        if variant=='aux':logits=p.risk_head(p._latents(tensor)[0]).numpy()
                    risk_mass.extend((probs*y).sum(1).tolist());uniform.extend(y.mean(1).tolist())
                    greedy.extend(y[np.arange(len(y)),probs.argmax(1)].tolist())
                    if variant=='aux':
                        acc.extend(balanced(logits,y).tolist());frozen_acc.extend(balanced(np.repeat(l0,len(y),axis=0),y).tolist());targets.append(y);predictions.append(logits)
                r={'representation':rep,'seed':seed,'variant':variant,'scenes':len(observations),'uniform_approx_danger_mass':float(np.mean(uniform)),'policy_approx_danger_mass':float(np.mean(risk_mass)),'greedy_approx_danger_fraction':float(np.mean(greedy))}
                if acc:
                    r.update(aux_balanced_accuracy=float(np.mean(acc)),frozen_aux_balanced_accuracy=float(np.mean(frozen_acc)))
                    y=np.concatenate(targets);z=np.concatenate(predictions);pos=y.sum(0);neg=len(y)-pos;ok=(pos>0)&(neg>0)
                    for name,logits in [('normal',z),('frozen',np.repeat(l0,len(y),axis=0))]:
                        score=.5*(((logits>=0)*y).sum(0)/np.maximum(pos,1)+((logits<0)*(1-y)).sum(0)/np.maximum(neg,1))
                        loss=.5*((np.logaddexp(0,-logits)*y).sum(0)/np.maximum(pos,1)+(np.logaddexp(0,logits)*(1-y)).sum(0)/np.maximum(neg,1))
                        r[name+'_action_balanced_accuracy']=float(score[ok].mean());r[name+'_action_balanced_bce']=float(loss[ok].mean())
                results.append(r);print(r,flush=True)
    (out/('probe-audit-'+('-'.join(args.representations) or 'all')+'.json')).write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()

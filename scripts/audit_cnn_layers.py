"""Read-only layer variation and matched observation interventions, CNN only."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.autumn_ablation import GridAblation
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_training import build_reference_model as build_model
from touhou_ai.simulation_speed import distributions

def collect(model):
    observations=[];metadata=[]
    for episode in range(12):
        env=GridAblation();o,_=env.reset(seed=4000+episode);history=[]
        while True:
            history.append((o,env.frame))
            a,_=model.predict(o,deterministic=False)
            o,_,done,trunc,_=env.step(a)
            if done or trunc:break
        ids=sorted(set([0,len(history)//3,2*len(history)//3]+list(range(max(0,len(history)-5),len(history)))))
        for i in ids:
            observations.append(history[i][0]);metadata.append(dict(episode=episode,frame=history[i][1],death=env.dead))
    return {k:np.stack([o[k] for o in observations]) for k in observations[0]},metadata

def outputs(policy,obs):
    ps=[];vs=[]
    with torch.no_grad():
        for i in range(0,len(obs['player']),8):
            part={k:v[i:i+8] for k,v in obs.items()};ds=distributions(policy,part)
            ps.append((ds[0].probs[:,:,None]*ds[2].probs[:,None,:]).reshape(-1,18).numpy())
            tensor,_=policy.obs_to_tensor(part);vs.append(policy.predict_values(tensor).numpy().ravel())
    return np.concatenate(ps),np.concatenate(vs)

def inspect(policy,obs):
    captured={};handles=[]
    def hook(name):
        def f(module,args,out):captured.setdefault(name,[]).append(out.detach().numpy().reshape(len(out),-1).copy())
        return f
    for side,extractor,mlp,head in [('actor',policy.pi_features_extractor,policy.mlp_extractor.policy_net,policy.action_net),('critic',policy.vf_features_extractor,policy.mlp_extractor.value_net,policy.value_net)]:
        for branch in ('local','global_scene','player','merge'):
            for i,layer in enumerate(getattr(extractor,branch)):
                if isinstance(layer,torch.nn.ReLU):handles.append(layer.register_forward_hook(hook(f'{side}.{branch}.{i}')))
        for i,layer in enumerate(mlp):
            if isinstance(layer,torch.nn.Tanh):handles.append(layer.register_forward_hook(hook(f'{side}.mlp.{i}')))
            if isinstance(layer,torch.nn.Linear):handles.append(layer.register_forward_hook(hook(f'{side}.mlp_linear.{i}')))
        handles.append(head.register_forward_hook(hook(side+'.head')))
    p,v=outputs(policy,obs)
    for h in handles:h.remove()
    stats={}
    for name,parts in captured.items():
        a=np.concatenate(parts);std=a.std(axis=0);rms=float(np.sqrt(np.mean(a*a)))
        stats[name]=dict(rms=rms,scene_std_rms=float(np.sqrt(np.mean(std*std))),variation_ratio=float(np.sqrt(np.mean(std*std))/(rms+1e-12)),
            constant_fraction=float(np.mean(std<1e-8)),zero_fraction=float(np.mean(a==0)))
        if '.mlp.' in name:
            stats[name].update(tanh_abs_above_099=float(np.mean(np.abs(a)>.99)),mean_tanh_derivative=float(np.mean(1-a*a)))
    experiments={}
    for name,keys in [('grid',['local_grid','global_grid']),('player',['player']),('all',list(obs))]:
        other={k:(np.roll(x,len(x)//2,axis=0) if k in keys else x) for k,x in obs.items()}
        q,w=outputs(policy,other)
        experiments[name]=dict(mean_policy_tv=float(np.abs(p-q).sum(axis=1).mean()/2),max_policy_tv=float(np.abs(p-q).sum(axis=1).max()/2),mean_value_change=float(np.abs(v-w).mean()),max_value_change=float(np.abs(v-w).max()))
    return dict(layers=stats,interventions=experiments,value_range=float(np.ptp(v)),policy_max_range=float(np.ptp(p,axis=0).max()))

def main():
    torch.set_num_threads(1);root=pathlib.Path('artifacts/cnn-layers-20260929');root.mkdir(exist_ok=True)
    for seed in (7,17,27):
        np.random.seed(17000+seed);torch.manual_seed(17000+seed)
        m=SeparateClipPPO.load(f'artifacts/autumn-separate-clip-20260928/cnn-switch-{seed}.zip',device='cpu')
        if '--reuse' in sys.argv:
            obs=dict(np.load(root/f'observations-{seed}.npz'))
            meta=json.loads((root/f'{seed}.json').read_text(encoding='utf-8'))['scenes']
        else:
            obs,meta=collect(m);np.savez_compressed(root/f'observations-{seed}.npz',**obs)
        trained=inspect(m.policy,obs);fresh,_,_=build_model('cnn',seed)
        result=dict(seed=seed,scenes=meta,input_variation={k:float(np.sqrt(np.mean(x.std(axis=0)**2))) for k,x in obs.items()},trained=trained,initial=inspect(fresh.policy,obs))
        (root/f'{seed}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(seed,'scenes',len(meta),'trained',trained['interventions'],'value range',trained['value_range'],flush=True)
if __name__=='__main__':main()

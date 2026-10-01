import sys,pathlib,time,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch,numpy as np
from stable_baselines3 import PPO
from touhou_ai.side_dodge import SideGrid,SideNumerical
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.evasion_only import EvasionPolicy
from scripts.continue_dodge_comparison import Progress

def assess_legacy(m,cls,swapped=False,n=50,det=False):
    result={};torch.manual_seed(123)
    for side in [-1,1]:
        e=cls();wins=0;xs=[]
        for _ in range(n):
            o,_=e.reset(options={'side':side})
            for step in range(80):
                a,_=m.predict(e.opposite_observation() if swapped else o,deterministic=det)
                o,r,d,_,info=e.step(a)
                if step==9:xs.append(float(e.pos[0]))
                if d:wins+=info['success'];break
        result[str(side)]={'survival':wins/n,'x_after20f':float(np.mean(xs))}
    return result

def assess(m,cls,swapped=False,n=50,det=False):
    from touhou_ai.simulation_speed import assess_batch
    return {str(side):assess_batch(m,cls,n=n,side=side,swapped=swapped,det=det) for side in [-1,1]}

def main():
    torch.set_num_threads(2);out=pathlib.Path('artifacts/side-warmstart-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();rows=[]
    for name,cls in [('numerical',SideNumerical),('narrow-cnn',SideGrid)]:
        kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
        if name=='narrow-cnn':kw['features_extractor_class']=NarrowGridFeatures
        m=PPO(EvasionPolicy,cls(),seed=7,device='cpu',n_steps=512,batch_size=64,n_epochs=3,learning_rate=.0001,gamma=.9995,gae_lambda=.95,target_kl=.02,ent_coef=.01,rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0)
        source=pathlib.Path('artifacts/side-classifier-20260928-173822')/(name+'.pt')
        learned=torch.load(source,map_location='cpu',weights_only=True)
        original_action={k:v.clone() for k,v in m.policy.action_net.state_dict().items()}
        original_value={k:v.clone() for k,v in m.policy.vf_features_extractor.state_dict().items()}
        m.policy.pi_features_extractor.load_state_dict(learned['encoder'],strict=True)
        m.policy.mlp_extractor.policy_net.load_state_dict(learned['actor_mlp'],strict=True)
        assert all(torch.equal(v,m.policy.action_net.state_dict()[k]) for k,v in original_action.items())
        assert all(torch.equal(v,m.policy.vf_features_extractor.state_dict()[k]) for k,v in original_value.items())
        assert len(m.policy.optimizer.state)==0
        from scripts.check_gap_classification import dataset
        from stable_baselines3.common.policies import BaseModel
        probe_data,probe_labels=dataset(cls,128,47)
        probe_weight=learned['classifier']['weight'];probe_bias=learned['classifier']['bias']
        def probe():
            predictions=[]
            with torch.no_grad():
                for i in range(0,len(probe_labels),64):
                    obs={k:torch.as_tensor(v[i:i+64],dtype=torch.float32) for k,v in probe_data.items()}
                    features=BaseModel.extract_features(m.policy,obs,m.policy.pi_features_extractor)
                    latent=m.policy.mlp_extractor.forward_actor(features)
                    predictions.extend(torch.nn.functional.linear(latent,probe_weight,probe_bias).argmax(1).tolist())
            return float(np.mean(np.array(predictions)==probe_labels))
        assert probe()==1.0
        from touhou_ai.simulation_speed import tune_cpu
        tune_cpu(m);start=time.perf_counter()
        for target in [0,8192,16384]:
            if target:m.learn(total_timesteps=target-m.num_timesteps,reset_num_timesteps=False,callback=Progress())
            ts=torch.get_rng_state();ns=np.random.get_state();row={'evaluation_protocol':'numpy-per-episode-123-batch16-v1','source_classifier':str(source),'actor_probe_accuracy':probe(),'model':name,'steps':m.num_timesteps,'normal':assess(m,cls),'greedy':assess(m,cls,n=1,det=True)}
            if target==16384:row['opposite_bullets']=assess(m,cls,True)
            torch.set_rng_state(ts);np.random.set_state(ns);row['seconds']=time.perf_counter()-start;rows.append(row);m.save(out/(name+'-'+str(target)));(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    print(out,flush=True)
if __name__=='__main__':main()

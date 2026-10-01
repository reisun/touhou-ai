"""Isolated DirectML preflight. No live game input or live learner changes."""
import sys,pathlib,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from stable_baselines3.common.logger import configure
from touhou_ai.live_action_grid import LiveActionGridContract,build_live_action_model,CONTRACT,SPEC
from touhou_ai.spatial_input_candidates import CandidateEnv
from touhou_ai.directml_update import update
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.model_monitor import model_metadata

out=pathlib.Path('artifacts/live-action-grid-preflight-20260929');out.mkdir(exist_ok=True)
torch.set_num_threads(1);profile=json.loads(pathlib.Path('configs/sharu-inspired-v1.json').read_text())
settings=profile['provisional_ppo']|profile['evasion_ppo_overrides']|{'n_steps':32,'batch_size':16}
model=build_live_action_model(settings);model.set_logger(configure(str(out),['json']));contract=LiveActionGridContract()
e=CandidateEnv();e.reset(seed=7);timings=[];observations=[]
for i in range(32):
    raw=e.raw_observation();raw['player']['position']=[i-16,330]
    raw['bullets']=[dict(position=[i-12,330],velocity_raw=[-1,0],hitbox_raw=[4,4],flags_raw=2)]
    if i%3==0:
        raw['lasers']=[{'collision':dict(origin=[i-8,320],angle=np.pi/2,length=20,width=2,active=True,field_validated=False)}]
    start=time.perf_counter();obs=contract.encode(raw);timings.append((time.perf_counter()-start)*1000);observations.append(obs)
    with torch.no_grad():
        tensor,_=model.policy.obs_to_tensor(obs);action,value,lp,ds=model.policy.forward_with_distribution(tensor)
        _,check,entropy=model.policy.evaluate_actions(tensor,action)
    torch.testing.assert_close(lp,check)
    assert action[0,1]==action[0,3]==0 and torch.isfinite(entropy).all()
    model.rollout_buffer.add(obs,action.numpy(),np.array([-60. if i==31 else 0.]),np.array([i==0]),value,lp)
model.rollout_buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]));model._current_progress_remaining=1.
before={k:v.clone() for k,v in model.policy.state_dict().items()}
model,info=update(model,model.rollout_buffer,out/'directml')
assert info['backend']=='directml',info
assert type(model) is SeparateClipPPO and any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items())
assert float(model.policy.pi_features_extractor.merge[0].weight[:,321:].detach().abs().sum())>0
model.save(out/'verified');loaded=SeparateClipPPO.load(out/'verified',env=contract,device='cpu')
for k,v in model.policy.state_dict().items():torch.testing.assert_close(v,loaded.policy.state_dict()[k],rtol=0,atol=0)
assert model_metadata(model,CONTRACT,None)['bullet_scope']==SPEC
np.testing.assert_array_equal(model.predict(observations[-1],deterministic=True)[0],loaded.predict(observations[-1],deterministic=True)[0])
result=dict(contract=CONTRACT,directml=info,parameters=sum(p.numel() for p in model.policy.parameters()),
    extra_actor_weights_updated=True,save_reload_equal=True,shot_bomb_disabled=True,
    encode_median_ms=float(np.median(timings)),encode_p95_ms=float(np.quantile(timings,.95)),
    actor_gradient_norm=float(model.logger.name_to_value['train/actor_gradient_norm']),critic_gradient_norm=float(model.logger.name_to_value['train/critic_gradient_norm']))
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

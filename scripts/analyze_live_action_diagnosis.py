"""Read-only snapshots, training metrics and common-scene checkpoint audit."""
import sys,pathlib,json,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.live_action_grid import LiveActionGridContract
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.live_runtime import input_mask
from scripts.audit_cnn_layers import outputs,inspect
ROOT=pathlib.Path('artifacts');OUT=ROOT/'live-action-diagnosis-20260929'
run=json.loads((OUT/'run.json').read_text())
current=json.loads((OUT/'current-status.json').read_text())
oldname='live-learning-20260929-071910-9a4a78'
old=json.loads((ROOT/oldname/'status.json').read_text())
def describe(status,rows,name):
    ts=[];metrics=[];initials=collections.Counter()
    for e in rows:
        initial=json.loads((ROOT/name/f"initial-state-{e['episode']}.json").read_text())
        initials[tuple(initial[k] for k in ('stage','stage_frame','difficulty','character','shot','lives_raw'))]+=1
        ts.append((e['final_frame']-initial['stage_frame'])/60)
        p=ROOT/name/f"directml-{e['episode']}"/'result.json'
        if p.exists():metrics.append(json.loads(p.read_text())['metrics'])
    keys=['train/entropy_loss','train/explained_variance','train/approx_kl','train/value_loss','train/actor_gradient_norm','train/critic_gradient_norm']
    return dict(first=rows[0]['episode'],last=rows[-1]['episode'],n=len(rows),seconds=float(np.mean(ts)),median_seconds=float(np.median(ts)),
        boss=sum((e.get('max_progress') or {}).get('rank',0)>=3 for e in rows),midboss=sum((e.get('max_progress') or {}).get('rank',0)>=1 for e in rows),
        kl_stops=sum(e['optimization'].get('kl_early_stopped',False) for e in rows),epochs=dict(collections.Counter(e['optimization']['completed_epochs'] for e in rows)),
        save_reload_all=all(e['parameters_changed'] and e['reload_verified'] for e in rows),
        truncated_or_recovery=sum(e['truncated'] or e['recovery_update'] for e in rows),
        initial_conditions={str(k):v for k,v in initials.items()},metrics={k:float(np.mean([m[k] for m in metrics if k in m])) for k in keys})
cap=current['updates'];steps=current['gameplay_training_steps']
windows={'current_last100':describe(current,current['episodes'][-100:],run['RunId']),
 'old_same_updates':describe(old,old['episodes'][cap-100:cap],oldname),
 'old_same_steps':describe(old,[e for e in old['episodes'] if e['total_steps']<=steps][-100:],oldname),
 'old_final100':describe(old,old['episodes'][-100:],oldname)}
blocks=[describe(current,current['episodes'][i:i+50],run['RunId']) for i in range(0,cap,50)]
telemetry=json.loads((OUT/'telemetry.json').read_text())
play=[d for d in telemetry if d['player'] and d['game']['lives_reserve']>=0]
probs=np.array([d['policy']['directions'] for d in play]);positions=np.array([d['player']['position'] for d in play])
hits=[]
for a,b in zip(play[:-1],play[1:]):
    if a['episode_id']==b['episode_id'] and b['game']['lives_reserve']<a['game']['lives_reserve']:
        hits.append(dict(episode=b['episode_id'],frame=b['game_frame'],before_position=a['player']['position'],before_bullets=len(a['entities']['bullets']),
                         before_enemies=len(a['entities']['enemies']),before_status=a['player']['status'],gap_frames=b['game_frame']-a['game_frame']))
observation=dict(samples=len(play),episodes=sorted(set(d['episode_id'] for d in play)),mean_direction_probs=probs.mean(0).tolist(),
    sampled_directions=dict(collections.Counter(d['policy']['action'][0] for d in play)),mean_focus=float(np.mean([d['policy']['focus'] for d in play])),
    mean_position=positions.mean(0).tolist(),fraction_top_y_below100=float(np.mean(positions[:,1]<100)),
    position_y_range=[float(positions[:,1].min()),float(positions[:,1].max())],
    action_input_mismatches=sum(d['applied_input']!=input_mask(d['policy']['action']) for d in play),
    hits=hits,value_mean=float(np.mean([d['policy']['value'] for d in play])),value_std=float(np.std([d['policy']['value'] for d in play])))
raws=json.loads((OUT/'raws.json').read_text());contract=LiveActionGridContract();obs=[contract.encode(r) for r in raws]
batch={k:np.stack([o[k] for o in obs]) for k in obs[0]}
torch.set_num_threads(1);checkpoints={}
selection=[('new',run['RunId'],i) for i in (1,25,50,100,200,cap)]+[('old',oldname,i) for i in (100,min(cap,566),566)]
for tag,name,index in selection:
    m=SeparateClipPPO.load(ROOT/name/f'real-episode-{index}.zip',device='cpu')
    inputs={k:v for k,v in batch.items() if k in m.observation_space.spaces}
    p,v=outputs(m.policy,inputs);layer=inspect(m.policy,inputs)
    direction=p.reshape(-1,9,2).sum(2);focus=p.reshape(-1,9,2).sum(1)
    row=dict(mean_direction_probs=direction.mean(0).tolist(),mean_focus=float(focus[:,1].mean()),value_mean=float(v.mean()),value_std=float(v.std()),
       critic_saturation=layer['layers']['critic.mlp.3']['tanh_abs_above_099'],actor_saturation=layer['layers']['actor.mlp.3']['tanh_abs_above_099'],
       grid_permutation_tv=layer['interventions']['grid']['mean_policy_tv'])
    if 'action_grid' in inputs:
        other={k:v.copy() for k,v in inputs.items()};other['action_grid'].fill(0);q,_=outputs(m.policy,other)
        row['risk_input_zero_tv']=float(np.abs(p-q).sum(1).mean()/2)
        row['risk_input_weight_norm']=float(torch.linalg.vector_norm(m.policy.pi_features_extractor.merge[0].weight[:,321:]).detach())
    checkpoints[f'{tag}-{index}']=row
result=dict(snapshot_updates=cap,snapshot_steps=steps,windows=windows,blocks=blocks,live_observation=observation,raw_scenes=len(raws),checkpoints=checkpoints)
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

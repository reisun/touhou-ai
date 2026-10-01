import sys,pathlib,json,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch,numpy as np
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import NumericalAblation

def norm(params):return float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in params if p.grad is not None)))
def main():
 torch.set_num_threads(1)
 m=PPO.load('artifacts/autumn-ablation-20260928/relative-switch-7.zip',env=NumericalAblation(relative=True),device='cpu')
 m.train=lambda:None;m.learn(512)
 rows=[]
 for i,b in enumerate(m.rollout_buffer.get(64)):
  variants={};deltas={}
  for mode in ['global_clip','separate_clip','actor_only']:
   p=copy.deepcopy(m.policy);p.set_training_mode(True)
   actor=list(p.pi_features_extractor.parameters())+list(p.mlp_extractor.policy_net.parameters())+list(p.action_net.parameters())
   critic=list(p.vf_features_extractor.parameters())+list(p.mlp_extractor.value_net.parameters())+list(p.value_net.parameters())
   assert not ({id(x) for x in actor}&{id(x) for x in critic})
   before=torch.cat([x.detach().flatten() for x in actor])
   values,logp,entropy=p.evaluate_actions(b.observations,b.actions)
   adv=b.advantages;adv=(adv-adv.mean())/(adv.std()+1e-8)
   ratio=torch.exp(logp-b.old_log_prob)
   policy_loss=-torch.min(adv*ratio,adv*torch.clamp(ratio,.8,1.2)).mean()-.01*entropy.mean()
   value_loss=.5*((values.flatten()-b.returns)**2).mean()
   p.optimizer.zero_grad();(policy_loss+(value_loss if mode!='actor_only' else 0)).backward()
   ag=norm(actor);cg=norm(critic) if mode!='actor_only' else 0.
   if mode=='separate_clip':
    torch.nn.utils.clip_grad_norm_(actor,.5);torch.nn.utils.clip_grad_norm_(critic,.5)
   else:torch.nn.utils.clip_grad_norm_(p.parameters(),.5)
   clipped=norm(actor);p.optimizer.step()
   delta=torch.cat([x.detach().flatten() for x in actor])-before;deltas[mode]=delta
   variants[mode]={'actor_gradient_norm':ag,'critic_gradient_norm':cg,'actor_gradient_after_clip':clipped,'actor_parameter_change':float(delta.norm())}
  a=deltas['global_clip'];bdelta=deltas['separate_clip']
  rows.append({'batch':i,'modes':variants,'global_vs_separate_update_cosine':float(torch.nn.functional.cosine_similarity(a,bdelta,dim=0))})
 report={'checkpoint':'relative-switch-7','rollout':'fresh512 on-policy, no training of source checkpoint','minibatches':len(rows),'rows':rows,'limits':'one-step copies retaining saved Adam state; gradient size alone does not measure Adam update; no evidence yet of long-run improvement'}
 out=pathlib.Path('artifacts/observation-credit-audit-20260928');(out/'clipping.json').write_text(json.dumps(report,indent=2))
 for mode in ['global_clip','separate_clip','actor_only']:
  print(mode,{k:float(np.mean([r['modes'][mode][k] for r in rows])) for k in rows[0]['modes'][mode]},flush=True)
 print('update_cosine',np.mean([r['global_vs_separate_update_cosine'] for r in rows]),flush=True)
if __name__=='__main__':main()

"""Offline two-frame action plans. Never imports live process/input control."""
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, BaseModel
from touhou_ai.spatial_input_candidates import CandidateEnv, CandidateFeatures
from touhou_ai.live_action_grid import LiveActionGridContract
from touhou_ai.fixed_dodge import DIRECTIONS

class PlanDistribution:
    def __init__(self, logits, split):
        self.first=Categorical(logits=logits[:,:9])
        self.second=Categorical(logits=logits[:,9:90].reshape(-1,9,9))
        self.focus=Categorical(logits=logits[:,90:92]);self.split=split
    def log_prob(self, actions):
        a,b,f=actions.long().unbind(1);idx=torch.arange(len(a),device=a.device)
        second=self.second.logits[idx,a].gather(1,b[:,None]).squeeze(1) if self.split else torch.where(a==b,0.,-1e9)
        return self.first.log_prob(a)+second+self.focus.log_prob(f)
    def entropy(self):
        return self.first.entropy()+self.focus.entropy()+(self.first.probs*self.second.entropy()).sum(1)*self.split
    def regularizer(self):
        # Same entropy encouragement per controlled frame; joint log_prob is NOT averaged.
        direction=self.first.entropy()
        if self.split:direction=.5*(direction+(self.first.probs*self.second.entropy()).sum(1))
        return direction+self.focus.entropy()
    def get_actions(self, deterministic=False):
        if deterministic and self.split:
            joint=self.first.logits[:,:,None]+self.second.logits
            ab=joint.flatten(1).argmax(1);a,b=ab//9,ab%9
        else:
            a=self.first.probs.argmax(1) if deterministic else self.first.sample()
            if self.split:
                dist=Categorical(logits=self.second.logits[torch.arange(len(a),device=a.device),a])
                b=dist.sample()
            else:b=a
        f=self.focus.probs.argmax(1) if deterministic else self.focus.sample()
        return torch.stack([a,b,f],1)

class PlanPolicy(MultiInputActorCriticPolicy):
    def __init__(self,*args,split=False,**kwargs):
        self.split=split
        super().__init__(*args,**kwargs)
        self.action_net=nn.Linear(self.mlp_extractor.latent_dim_pi,92)
        self.optimizer=self.optimizer_class(self.parameters(),lr=self.optimizer.param_groups[0]['lr'],**self.optimizer_kwargs)
    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters();data['split']=self.split;return data
    def latents(self,obs):
        pi,vf=self.extract_features(obs)
        return self.mlp_extractor.forward_actor(pi),self.mlp_extractor.forward_critic(vf)
    def get_distribution(self,obs):
        f=BaseModel.extract_features(self,obs,self.pi_features_extractor)
        return PlanDistribution(self.action_net(self.mlp_extractor.forward_actor(f)),self.split)
    def forward(self,obs,deterministic=False):
        pi,vf=self.latents(obs);dist=PlanDistribution(self.action_net(pi),self.split)
        a=dist.get_actions(deterministic);return a,self.value_net(vf),dist.log_prob(a)
    def evaluate_actions(self,obs,actions):
        pi,vf=self.latents(obs);dist=PlanDistribution(self.action_net(pi),self.split)
        return self.value_net(vf),dist.log_prob(actions),dist.regularizer()

class FrameVibration:
    alpha=1-np.sqrt(.75)
    weights=np.zeros((36,36))
    for i in range(36):
        weights[i,0]=(1-alpha)**i
        for j in range(1,i+1):weights[i,j]=alpha*(1-alpha)**(i-j)
    def __init__(self):
        from collections import deque
        self.moves=deque(maxlen=36);self.score=0.
    def add(self,delta):
        # Convert 1F displacements to px/2F units and retain the same 36F time horizon.
        v=np.asarray(delta)*2.;previous=self.moves[-1] if self.moves else None;self.moves.append(v)
        if len(self.moves)<36:return 0.
        history=np.asarray(self.moves)
        residual=np.linalg.norm(history-self.weights@history,axis=1).mean()
        acceleration=np.diff(history,axis=0);norms=np.linalg.norm(acceleration,axis=1)
        active=acceleration[norms>.01];norms=norms[norms>.01]
        alternating=float(np.maximum(0.,-np.sum(active[1:]*active[:-1],axis=1)/(norms[1:]*norms[:-1])).sum())
        self.score=float(residual*min(1.,max(0.,(alternating-1)/3)))
        if previous is None or np.linalg.norm(v)<=.01 or np.linalg.norm(v-previous)<=.01:return 0.
        return .1*min(1.,self.score/4)/60

class PlanEnv(CandidateEnv):
    def __init__(self,start_y=330,split=False):
        self.start_y=start_y;self.split=split;self.previous={};self.vibration=FrameVibration()
        super().__init__('action_grid');self.encoder=LiveActionGridContract();self.observation_space=self.encoder.observation_space
        self.action_space=gym.spaces.MultiDiscrete([9,9,2])
    def reset(self,**kw):
        self.previous={};self.vibration=FrameVibration()
        super().reset(**kw);self.pos[1]=self.start_y
        return self.observe(),{}
    def observe(self):
        if not isinstance(self.encoder,LiveActionGridContract):return super().observe()
        obs=self.encoder.encode(self.raw_observation(),self.previous)
        obs['bomb_clock']=np.array([(self.frame%12)/12],np.float32);return obs
    def step(self,action):
        a,b,focus=map(int,action)
        if not self.split and a!=b:raise ValueError('held arm must repeat first direction')
        if self.dead or self.frame>=self.horizon:raise RuntimeError('reset required')
        self.focus=focus;deltas=[];scores=[];penalty=0.;valid_frames=[]
        for direction in (a,b):
            before=self.pos.copy();pv=DIRECTIONS[direction]*(2 if focus else 4.5)
            if self.frame and self.frame%120==0:self._emit()
            ids,collidable=self._frame_velocity()
            for _ in range(4):
                self.pos=np.clip(self.pos+pv*.25,[-184,32],[184,432]);self.xy[ids]+=self.vel[ids]*.25
                if np.any(np.all(np.abs(self.xy[ids[collidable]]-self.pos)<=3,axis=1)):self.dead=True;break
            self._cull();t=self.frame%120
            if t<60:
                self.boss+=np.array([np.cos(self.move_angle),np.sin(self.move_angle)])*2*(1-t/60)**2
                self.boss=np.clip(self.boss,[-140,46],[140,146])
            self.frame+=1;delta=self.pos-before;deltas.append(delta.tolist())
            valid_frames.append(not self.dead)
            if self.dead:break
        # Only position capture/action switching occurs at the intermediate frame.
        # Reward/diagnostic work is batched after both scheduled controls.
        for delta,valid in zip(deltas,valid_frames):
            if valid:penalty+=self.vibration.add(delta)
            scores.append(self.vibration.score)
        reward=-1. if self.dead else 0.;self.previous={'hit':reward}
        done=self.dead or self.frame>=self.horizon
        return self.observe(),reward-penalty,done,False,dict(success=done and not self.dead,frames=self.frame,deltas=deltas,vibration=scores,penalty=penalty)

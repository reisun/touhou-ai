"""Offline candidate observations derived only from currently observable raw data.

No simulator future branching, action masks, additional rewards or live changes.
"""
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from touhou_ai.dual_grid import DualGridContract,rectangle,paint_laser
from touhou_ai.autumn_ablation import GridAblation
from touhou_ai.fixed_dodge import DIRECTIONS
from touhou_ai.narrow_grid import NarrowGridFeatures

VARIANTS=('pixel1','action_grid','geometry')
TILES=[(1,1),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(0,0)]
LOW=np.array([-184.,32.]);HIGH=np.array([184.,432.])

def build_candidate_model(variant,seed,baseline,config):
    """Match the tested initialization, including neutral added connections."""
    import functools
    from touhou_ai.separate_clip_ppo import SeparateClipPPO
    from touhou_ai.autumn_ablation import SpeedPolicy
    from touhou_ai.dual_grid import GridRolloutBuffer
    rng=torch.get_rng_state();nrng=np.random.get_state()
    cls=functools.partial(CandidateEnv,variant=variant)
    model=SeparateClipPPO(SpeedPolicy,cls(),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,
        policy_kwargs=dict(share_features_extractor=False,net_arch=dict(pi=[256,128],vf=[256,128]),features_extractor_class=CandidateFeatures),verbose=0,**config['ppo'])
    old=baseline.policy.state_dict()
    with torch.no_grad():
        for key,target in model.policy.state_dict().items():
            if key not in old:continue
            source=old[key]
            if target.shape==source.shape:target.copy_(source)
            else:
                assert target.ndim>=2 and target.shape[0]==source.shape[0] and target.shape[2:]==source.shape[2:]
                assert target.shape[1]>source.shape[1]
                target.zero_();target[:,:source.shape[1]].copy_(source)
    torch.set_rng_state(rng);np.random.set_state(nrng)
    return model,cls

def paint(layer,positions,half_sizes,origin,cell):
    """Exact per-object cell coverage; maximum, as in the baseline encoder."""
    n=layer.shape[0]
    lo=(positions-half_sizes-origin)/cell;hi=(positions+half_sizes-origin)/cell
    starts=np.clip(np.floor(lo),0,n).astype(int);ends=np.clip(np.ceil(hi),0,n).astype(int)
    widths=ends-starts;areas=np.prod(widths,axis=1);selected=np.flatnonzero(areas>0)
    for offset in range(0,len(selected),128):
        group=selected[offset:offset+128];counts=areas[group];idx=np.repeat(group,counts)
        ordinal=np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts,counts)
        x=starts[idx,0]+ordinal%widths[idx,0];y=starts[idx,1]+ordinal//widths[idx,0]
        dx=np.maximum(0,np.minimum(x+1,hi[idx,0])-np.maximum(x,lo[idx,0]))
        dy=np.maximum(0,np.minimum(y+1,hi[idx,1])-np.maximum(y,lo[idx,1]))
        np.maximum.at(layer,(y,x),(dx*dy).astype(np.float32))

def bullet_arrays(raw):
    active=[b for b in raw['bullets'] if b['flags_raw']&2]
    return tuple(np.array([b[k] for b in active],float).reshape(-1,2) for k in ('position','velocity_raw','hitbox_raw'))

def risk_grid(raw):
    """4x3x3: fast2F, slow2F, fast16F, slow16F; move2F then stop.

    Continuous AABB segment intersections, including wall clipping. Constant
    observed bullet velocity; no future turns or new bullets are known.
    """
    if raw['lasers']:raise ValueError('Action-risk candidate currently supports bullet-only scenes')
    origin=np.array(raw['player']['position'],float);half=np.array(raw['player']['hitbox_raw'],float)
    p,v,size=bullet_arrays(raw);extent=size/2+half
    near=np.all(np.abs(p-origin)<=np.abs(v)*16+extent+10,axis=1)
    p,v,extent=p[near],v[near],extent[near]
    out=np.zeros((4,3,3),np.float32)
    if not len(p):return out
    for focus,speed in enumerate((4.5,2.)):
        for direction,(iy,ix) in enumerate(TILES):
            velocity=DIRECTIONS[direction]*speed
            # Each interval has linear player/bullet motion. Include exact wall-hit times.
            times=[0.,2.,16.]
            for axis in (0,1):
                if velocity[axis]:
                    t=((HIGH if velocity[axis]>0 else LOW)[axis]-origin[axis])/velocity[axis]
                    if 0<t<2:times.append(t)
            times=sorted(times)
            for a,b in zip(times[:-1],times[1:]):
                x0=np.clip(origin+min(a,2)*velocity,LOW,HIGH)
                x1=np.clip(origin+min(b,2)*velocity,LOW,HIGH)
                rel=p+a*v-x0;delta=(b-a)*v-(x1-x0)
                stationary=np.abs(delta)<1e-12
                den=np.where(stationary,1,delta)
                u0=(-extent-rel)/den;u1=(extent-rel)/den
                enter=np.where(stationary,-np.inf,np.minimum(u0,u1)).max(axis=1)
                leave=np.where(stationary,np.inf,np.maximum(u0,u1)).min(axis=1)
                valid=~np.any(stationary&(np.abs(rel)>extent),axis=1)
                hit=valid&(np.maximum(enter,0)<=np.minimum(leave,1))
                if hit.any():
                    out[2+focus,iy,ix]=1
                    if b<=2:out[focus,iy,ix]=1
    return out

class CandidateEncoder:
    def __init__(self,variant):
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.base=DualGridContract()
        spaces=dict(self.base.observation_space.spaces)
        if variant=='pixel1':spaces['local_grid']=gym.spaces.Box(-1,1,(6,192,192),dtype=np.float32)
        if variant=='geometry':spaces['local_grid']=gym.spaces.Box(-1,1,(13,96,96),dtype=np.float32)
        if variant=='action_grid':spaces['action_grid']=gym.spaces.Box(0,1,(4,3,3),dtype=np.float32)
        self.observation_space=gym.spaces.Dict(spaces)
    def encode(self,raw):
        out=self.base.encode(raw)
        pos=np.array(raw['player']['position'],float);half=np.array(raw['player']['hitbox_raw'],float)
        origin=pos-96;p,v,size=bullet_arrays(raw)
        if self.variant=='pixel1':
            grid=np.zeros((6,192,192),np.float32)
            paint(grid[0],pos[None,:],half[None,:],origin,1)
            for channel,t in enumerate((0,2,4),1):paint(grid[channel],p+t*v,size/2,origin,1)
            for laser in raw['lasers']:paint_laser(grid,4,5,laser['collision'],origin,1)
            out['local_grid']=grid
        elif self.variant=='geometry':
            extra=np.zeros((7,96,96),np.float32)
            for channel,t in enumerate((0,2,4)):paint(extra[channel],p+t*v,size/2+half,origin,2)
            # Expanded danger regions are queried by PLAYER CENTER, not another expanded hitbox.
            # Unit-square center markers; separate endpoint and swept-path planes.
            for focus,speed in enumerate((4.5,2.)):
                velocity=DIRECTIONS*speed
                end=np.clip(pos+2*velocity,LOW,HIGH)
                paint(extra[3+focus],end,np.full((9,2),.5),origin,2)
                path=np.clip(pos+np.linspace(0,2,17)[:,None,None]*velocity[None,:,:],LOW,HIGH).reshape(-1,2)
                paint(extra[5+focus],path,np.full_like(path,.5),origin,2)
            out['local_grid']=np.concatenate([out['local_grid'],extra])
        else:out['action_grid']=risk_grid(raw)
        for key,array in out.items():
            if key.endswith('_grid'):out[key]=array.astype(np.float16).astype(np.float32)
        return out

class CandidateEnv(GridAblation):
    def __init__(self,variant='pixel1'):
        super().__init__();self.encoder=CandidateEncoder(variant);self.observation_space=self.encoder.observation_space
    def raw_observation(self):
        # Same current-state fields as AutumnSky.observe; no hidden age, future schedule or turns.
        ids=np.flatnonzero(self.alive)
        return {'stage':1,'stage_frame':self.frame,'lives_raw':0,'power_raw':0,
            'player':{'position':self.pos.tolist(),'velocity_raw':[0,0],'hitbox_raw':[1,1],'status':4 if self.dead else 1,'invincibility_raw':0,'focus_raw':self.focus},
            'bullets':[{'position':p.tolist(),'velocity_raw':v.tolist(),'hitbox_raw':[4.,4.],'flags_raw':2} for p,v in zip(self.xy[ids],self.vel[ids])],
            'enemies':[],'items':[],'lasers':[],'player_shots':[],'bomb':{'state':0},'spell':None}
    def observe(self):return self.encoder.encode(self.raw_observation())

class CandidateFeatures(NarrowGridFeatures):
    def __init__(self,observation_space):
        super().__init__(observation_space)
        self.has_action='action_grid' in observation_space.spaces
        if observation_space['local_grid'].shape[-1]==192:
            # Retain parameter budget and 128-feature interface; process 1px detail BEFORE pooling.
            self.local[8]=nn.Sequential(nn.AdaptiveAvgPool2d((12,12)),nn.Flatten())
        channels=observation_space['local_grid'].shape[0]
        if channels!=6:self.local[0]=nn.Conv2d(channels,4,3,padding=1)
        if self.has_action:
            self.action=nn.Sequential(nn.Conv2d(4,8,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(72,32),nn.ReLU())
            self.merge[0]=nn.Linear(353,256)
    def forward(self,obs):
        parts=[self.local(obs['local_grid']),self.global_scene(obs['global_grid']),self.player(obs['player']),self.reward(obs['previous_rewards']),obs['bomb_clock']]
        if self.has_action:parts.append(self.action(obs['action_grid']))
        return self.merge(torch.cat(parts,dim=1))

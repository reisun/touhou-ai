"""Real-game observation extension; observed geometry only, no action override."""
import numpy as np
import gymnasium as gym
import torch
from touhou_ai.dual_grid import DualGridContract,SPEC as BASE_SPEC,GridRolloutBuffer
from touhou_ai.spatial_input_candidates import risk_grid,TILES,LOW,HIGH,CandidateFeatures
from touhou_ai.fixed_dodge import DIRECTIONS

CONTRACT='th10-dual-grid-v6-action-grid-v1'
SPEC=BASE_SPEC|{'action_grid':{'version':1,'shape':[4,3,3],
    'channels':['fast_2f','slow_2f','fast_16f','slow_16f'],
    'path':'move 2F then stop; clipped to player bounds',
    'bullets':'constant observed velocity; continuous AABB intersections',
    'lasers':'current active oriented rectangle held static; continuous AABB/OBB SAT',
    'speeds':[4.5,2.0],'bounds':[LOW.tolist(),HIGH.tolist()],
    'meaning':'geometric contact, including during invincibility; not guaranteed future hit'}}

def swept_laser_contact(start,end,half,collision):
    """Continuous separating-axis intersection of moving AABB and static OBB."""
    angle=float(collision['angle']);length=float(collision['length']);width=float(collision['width'])
    u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
    center=np.asarray(collision['origin'],float)+u*length/2
    axes=np.array([[1.,0.],[0.,1.],u,v])
    extent=np.abs(axes)@half+np.abs(axes@u)*length/2+np.abs(axes@v)*width/2
    p=axes@(start-center);delta=axes@(end-start)
    stationary=np.abs(delta)<1e-12
    if np.any(stationary&(np.abs(p)>extent)):return False
    denominator=np.where(stationary,1.,delta)
    left=(-extent-p)/denominator;right=(extent-p)/denominator
    enter=np.where(stationary,-np.inf,np.minimum(left,right)).max()
    leave=np.where(stationary,np.inf,np.maximum(left,right)).min()
    return bool(max(enter,0)<=min(leave,1))

def live_risk_grid(raw):
    # The simulator predictor deliberately rejects lasers; add their geometry here.
    out=risk_grid(raw|{'lasers':[]})
    lasers=[l['collision'] for l in raw['lasers'] if l['collision']['active']]
    if not lasers:return out
    start=np.asarray(raw['player']['position'],float);half=np.asarray(raw['player']['hitbox_raw'],float)
    # Prepare each rectangle once, then test all movement segments together.
    # Keep the scalar geometry arithmetic identical to swept_laser_contact.
    axes_list=[];centers=[];extents=[]
    for c in lasers:
        angle=float(c['angle']);length=float(c['length']);width=float(c['width'])
        u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
        axes=np.array([[1.,0.],[0.,1.],u,v])
        axes_list.append(axes)
        centers.append(np.asarray(c['origin'],float)+u*length/2)
        extents.append(np.abs(axes)@half+np.abs(axes@u)*length/2+np.abs(axes@v)*width/2)
    starts=[];ends=[];owners=[]
    for focus,speed in enumerate((4.5,2.)):
        for direction,(iy,ix) in enumerate(TILES):
            velocity=DIRECTIONS[direction]*speed;times=[0.,2.]
            for axis in (0,1):
                if velocity[axis]:
                    t=((HIGH if velocity[axis]>0 else LOW)[axis]-start[axis])/velocity[axis]
                    if 0<t<2:times.append(t)
            times=sorted(times)
            for a,b in zip(times[:-1],times[1:]):
                p=np.clip(start+a*velocity,LOW,HIGH);q=np.clip(start+b*velocity,LOW,HIGH)
                starts.append(p);ends.append(q);owners.append((focus,iy,ix))
    axes=np.asarray(axes_list)[None,:,:,:]
    starts=np.asarray(starts);ends=np.asarray(ends)
    p=(axes@(starts[:,None,:]-np.asarray(centers))[...,None])[...,0]
    delta=(axes@(ends-starts)[:,None,:,None])[...,0]
    extent=np.asarray(extents)[None,:,:]
    stationary=np.abs(delta)<1e-12
    outside=np.any(stationary&(np.abs(p)>extent),axis=-1)
    denominator=np.where(stationary,1.,delta)
    left=(-extent-p)/denominator;right=(extent-p)/denominator
    enter=np.where(stationary,-np.inf,np.minimum(left,right)).max(axis=-1)
    leave=np.where(stationary,np.inf,np.maximum(left,right)).min(axis=-1)
    hits=np.any(~outside&(np.maximum(enter,0)<=np.minimum(leave,1)),axis=-1)
    for hit,(focus,iy,ix) in zip(hits,owners):
        if hit:out[focus,iy,ix]=out[focus+2,iy,ix]=1
    return out

class LiveActionGridContract(DualGridContract):
    def __init__(self):
        super().__init__()
        self.observation_space=gym.spaces.Dict(dict(self.observation_space.spaces)|{
            'action_grid':gym.spaces.Box(0,1,(4,3,3),dtype=np.float32)})
    def encode(self,raw,previous_rewards=None):
        out=super().encode(raw,previous_rewards)
        out['action_grid']=live_risk_grid(raw).astype(np.float16).astype(np.float32)
        return out

def build_live_action_model(settings, evasion_only=True):
    """Fresh weights; same neutral-extension initialization as the adopted trial."""
    from touhou_ai.evasion_only import EvasionPolicy
    from touhou_ai.timed_bomb import TimedBombPolicy
    policy_class = EvasionPolicy if evasion_only else TimedBombPolicy
    from touhou_ai.narrow_grid import NarrowGridFeatures
    from touhou_ai.separate_clip_ppo import SeparateClipPPO
    kwargs=dict(device='cpu',rollout_buffer_class=GridRolloutBuffer,**settings)
    policy=dict(share_features_extractor=False,net_arch=dict(pi=[256,128],vf=[256,128]))
    base=SeparateClipPPO(policy_class,DualGridContract(),policy_kwargs=policy|{'features_extractor_class':NarrowGridFeatures},**kwargs)
    rng=torch.get_rng_state();nrng=np.random.get_state()
    model=SeparateClipPPO(policy_class,LiveActionGridContract(),policy_kwargs=policy|{'features_extractor_class':CandidateFeatures},**kwargs)
    old=base.policy.state_dict()
    with torch.no_grad():
        for key,target in model.policy.state_dict().items():
            if key not in old:continue
            source=old[key]
            if target.shape==source.shape:target.copy_(source)
            else:
                assert target.ndim==2 and target.shape[0]==source.shape[0] and target.shape[1]>source.shape[1]
                target.zero_();target[:,:source.shape[1]].copy_(source)
    torch.set_rng_state(rng);np.random.set_state(nrng)
    return model

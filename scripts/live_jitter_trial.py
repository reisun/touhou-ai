"""Offline simulator trials initialized from a fixed real-game checkpoint."""
import sys,pathlib,json,functools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from touhou_ai.scaled_candidate import ScaledCandidateEnv
from touhou_ai.live_action_grid import LiveActionGridContract
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.evasion_only import EvasionPolicy
from touhou_ai.spatial_input_candidates import CandidateFeatures
from touhou_ai.dual_grid import GridRolloutBuffer
from touhou_ai.simulation_speed import distributions,tune_cpu
from scripts.audit_motion_jitter import metrics
OUT=pathlib.Path('artifacts/live-jitter-trial-20261001')

class Env(ScaledCandidateEnv):
    def __init__(self,penalty=0):
        self.penalty=penalty;self.previous={};self.moves=[];self.cooldown=0
        super().__init__('action_grid',-1);self.encoder=LiveActionGridContract();self.observation_space=self.encoder.observation_space
    def reset(self,**kw):
        self.previous={};self.moves=[];self.cooldown=0
        return super().reset(**kw)
    def observe(self):
        if not isinstance(self.encoder,LiveActionGridContract):return super().observe()
        obs=self.encoder.encode(self.raw_observation(),self.previous)
        obs['bomb_clock']=np.array([(self.frame%12)/12],np.float32)
        return obs
    def step(self,a):
        p=self.pos.copy();_,r,d,tr,info=super().step(a)
        self.moves.append(self.pos-p);self.moves=self.moves[-6:];self.cooldown=max(0,self.cooldown-1)
        flagged=False
        if len(self.moves)==6 and not self.cooldown:
            v=np.array(self.moves);norm=np.linalg.norm(v,axis=1);length=norm.sum()
            rev=(np.sum(v[1:]*v[:-1],axis=1)<-.5*norm[1:]*norm[:-1])&(norm[1:]>.01)&(norm[:-1]>.01)
            flagged=bool(length>=1 and 1-np.linalg.norm(v.sum(0))/length>=.75 and rev.sum()>=2)
            if flagged:self.cooldown=6
        self.previous={'hit':r}
        info=dict(info,jitter=flagged,penalty=self.penalty*flagged)
        return self.observe(),r-self.penalty*flagged,d,tr,info

def evaluate(m,n=32,hysteresis=None,start_y=None):
    results=[]
    for offset in range(0,n,8):
        envs=[Env() for _ in range(min(8,n-offset))];obs=[e.reset(seed=5000+offset+i)[0] for i,e in enumerate(envs)]
        if start_y is not None:
            for i,e in enumerate(envs):e.pos[1]=start_y;obs[i]=e.observe()
        rng=[np.random.default_rng(19000+offset+i) for i in range(len(envs))];traces=[[] for _ in envs];active=list(range(len(envs)));last=[None]*len(envs)
        for t in range(300):
            batch={k:np.stack([obs[i][k] for i in active]) for k in obs[0]}
            with torch.inference_mode():probs=[d.probs.cpu().numpy() for d in distributions(m.policy,batch)]
            follow=[]
            for j,i in enumerate(active):
                u=rng[i].random(4);a=[min(int(np.searchsorted(np.cumsum(p[j]),u[h])),p.shape[1]-1) for h,p in enumerate(probs)]
                best=int(probs[0][j].argmax())
                if hysteresis is not None:
                    a[0]=best
                    if last[i] is not None and probs[0][j,best]-probs[0][j,last[i]]<=hysteresis:a[0]=last[i]
                    assert probs[0][j,best]-probs[0][j,a[0]]<=hysteresis+1e-7
                last[i]=a[0];before=envs[i].pos.copy();obs[i],_,done,_,info=envs[i].step(a)
                pp=np.sort(probs[0][j]);traces[i].append(dict(best=best,margin=float(pp[-1]-pp[-2]),action=a,delta=(envs[i].pos-before).tolist()))
                if done:results.append(dict(seed=5000+offset+i,success=bool(info['success']),frames=info['frames'],metrics=metrics(traces[i]),trace=traces[i]))
                else:follow.append(i)
            active=follow
            if not active:break
        assert not active
        print('eval',hysteresis,offset+len(envs),flush=True)
    return results

def build(seed,penalty):
    source=json.loads((OUT/'manifest.json').read_text())['checkpoint']
    old=SeparateClipPPO.load(source,device='cpu')
    settings=json.loads(pathlib.Path('configs/autumn-learning-baseline.json').read_text())['ppo']
    m=SeparateClipPPO(EvasionPolicy,Env(penalty),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,policy_kwargs=dict(share_features_extractor=False,net_arch=dict(pi=[256,128],vf=[256,128]),features_extractor_class=CandidateFeatures),verbose=0,**settings)
    assert m.observation_space==old.observation_space
    m.policy.load_state_dict(old.policy.state_dict(),strict=True)
    assert all(torch.equal(v,old.policy.state_dict()[k]) for k,v in m.policy.state_dict().items())
    return m

if __name__=='__main__':
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    mode=sys.argv[1]
    if mode=='probe':
        record=json.loads(pathlib.Path('.runtime/live-learning.json').read_text());folder=pathlib.Path(record['Output']);s=json.loads((folder/'status.json').read_text())
        if not (OUT/'manifest.json').exists():(OUT/'manifest.json').write_text(json.dumps(dict(checkpoint=str(folder/s['episodes'][-1]['checkpoint']),updates=s['episodes'][-1]['total_updates'],note='real weights and encoder; simulator bullet-only physics; shot/bomb masked; fine-tuning uses fresh optimizer, identical across arms'),indent=2))
        m=build(7,0);r=evaluate(m,16);(OUT/'probe.json').write_text(json.dumps(r))
        print('argmax switches',sum(x['metrics']['argmax_switches'] for x in r),flush=True)
    elif mode=='system':
        m=build(7,0)
        for h in [None,0,.05,.15]:
            r=evaluate(m,hysteresis=h);(OUT/f'system-{h}.json').write_text(json.dumps(r))
    else:
        if mode=='sweep' and (OUT/'saturation-stop.json').exists():raise RuntimeError('Saturation guard stopped further penalty increases')
        seed=int(sys.argv[2]);penalty=float(sys.argv[3]);m=build(seed,penalty);tune_cpu(m,4)
        assert 0<=penalty<=1
        from scripts.compare_critic_tanh import Record
        dest=OUT/f'train-{seed}-{penalty}';dest.mkdir(exist_ok=True)
        if mode=='sweep':
            from scripts.jitter_saturation import SaturationRecord
            callback=SaturationRecord(dest/'learning.json')
        else:callback=Record(dest/'learning.json')
        m.learn(8192,callback=callback);m.save(dest/'model.zip')
        reloaded=SeparateClipPPO.load(dest/'model.zip',device='cpu')
        assert all(torch.equal(v,reloaded.policy.state_dict()[k]) and torch.isfinite(v).all() for k,v in m.policy.state_dict().items())
        r=evaluate(m);(dest/'evaluation.json').write_text(json.dumps(r))
        if mode=='sweep':
            r=evaluate(m,start_y=160);(dest/'challenge.json').write_text(json.dumps(r))

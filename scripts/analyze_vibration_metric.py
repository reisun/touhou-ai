"""Offline diagnostic only: causal residual movement with repeated-change weighting."""
import json, math, pathlib, collections, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from touhou_ai.motion_jitter import MotionJitter
OUT=ROOT/'artifacts/vibration-metric-20261002'

def metric(moves, alpha=.25):
    v=np.asarray(moves,dtype=float)
    smooth=v[0].copy();res=[]
    for x in v:
        smooth += alpha*(x-smooth)
        res.append(float(np.linalg.norm(x-smooth)))
    # Ignore unchanged velocity, but keep the entire calculation within 36F.
    acceleration=np.diff(v,axis=0)
    active=[a for a in acceleration if np.linalg.norm(a)>.01]
    alternating=sum(max(0.,-float(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                    for a,b in zip(active,active[1:]))
    repeat_weight=min(1.,max(0.,(alternating-1.)/3.))
    residual=float(np.mean(res))
    return dict(residual_px_per_decision=residual,alternating_change=alternating,
                repeat_weight=repeat_weight,vibration=residual*repeat_weight,
                mean_speed=float(np.linalg.norm(v,axis=1).mean()))

def fixtures():
    paths={'straight':[(4,0)]*18,'stationary':[(0,0)]*18,
      'single_turn':[(4,0)]*9+[(-4,0)]*9,
      'single_corner':[(4,0)]*9+[(0,4)]*9,
      'single_stop_start':[(4,0)]*6+[(0,0)]*6+[(4,0)]*6,
      'single_speed_change':[(4,0)]*9+[(9,0)]*9,
      'alternating':[(4,0),(-4,0)]*9,
      'pause_between':([(4,0),(0,0),(-4,0),(0,0)]*5)[:18],
      'forward_zigzag':[(4,4),(4,-4)]*9,
      'pulse_forward':[(4,0),(0,0)]*9,
      'smooth_quarter_circle':[(4*math.cos(t),4*math.sin(t)) for t in np.linspace(0,math.pi/2,18)]}
    out={}
    for name,moves in paths.items():
        d=MotionJitter();out[name]=dict(metric(moves),v2_events=sum(d.add(*v) for v in moves))
    return out

def analyze():
    rows=json.loads((OUT/'capture.json').read_text());q=collections.deque(maxlen=18);prev=None;windows=[];invalid=collections.Counter();detector=MotionJitter();flags=collections.deque(maxlen=18)
    for r in rows:
        reason=None
        if prev is None or r['episode_id']!=prev['episode_id']:reason='episode_boundary'
        elif r['game_frame']-prev['game_frame']!=2:reason='missing_2f'
        elif any(x['player']['status']!=1 for x in [prev,r]):reason='player_state'
        elif r['game']['stage']!=prev['game']['stage'] or r['game']['lives_reserve']<prev['game']['lives_reserve']:reason='death_or_stage'
        elif any(x['game']['mode_flags'] not in (0,4) for x in [prev,r]):reason='menu'
        if reason:
            invalid[reason]+=1;q.clear();flags.clear();detector.reset()
        else:
            v=np.subtract(r['player']['position'],prev['player']['position'])
            if not np.isfinite(v).all() or np.linalg.norm(v)>16:
                invalid['discontinuous_position']+=1;q.clear();flags.clear();detector.reset()
            else:
                flag=detector.add(*v);q.append(v);flags.append(flag)
                if len(q)==18:
                    m=metric(q);windows.append(dict(m,episode=r['episode_id'],frame=r['game_frame'],v2_event=flag,v2_event_in_window=any(flags),
                      logged_jitter=r['reward']['components'].get('jitter',0),moves=[x.tolist() for x in q]))
        prev=r
    values=np.array([w['vibration'] for w in windows]);residuals=np.array([w['residual_px_per_decision'] for w in windows])
    summary=dict(samples=len(rows),episodes=dict(collections.Counter(r['episode_id'] for r in rows)),
      valid_windows=len(windows),invalid=dict(invalid),quantiles=dict(zip(['p0','p25','p50','p75','p90','p95','p100'],np.quantile(values,[0,.25,.5,.75,.9,.95,1]).tolist())),
      pure_residual_median=float(np.median(residuals)),positive_windows=int(sum(values>1e-6)),
      logged_jitter_events=sum(r['reward']['components'].get('jitter',0)<0 for r in rows),
      diagnostic_above_1_windows=int(sum(values>=1)),diagnostic_above_1_without_window_event=int(sum(w['vibration']>=1 and not w['v2_event_in_window'] for w in windows)))
    # Disjoint representative windows, so adjacent copies are not presented as separate cases.
    examples=[]
    for w in sorted(windows,key=lambda w:w['vibration'],reverse=True):
        if all(w['episode']!=x['episode'] or abs(w['frame']-x['frame'])>=36 for x in examples):examples.append(w)
        if len(examples)==8:break
    (OUT/'windows.json').write_text(json.dumps(windows));(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'examples.json').write_text(json.dumps(examples,indent=2))
    print(json.dumps(summary,indent=2))
    print('examples',json.dumps(examples[:3]))
    # Robustness check: smoothing coefficients are diagnostic, not fitted rewards.
    sensitivity={}
    for alpha in [.15,.25,.4]:
        scores=np.array([metric(w['moves'],alpha)['vibration'] for w in windows])
        sensitivity[str(alpha)]=dict(median=float(np.median(scores)),p90=float(np.quantile(scores,.9)),correlation_to_default=float(np.corrcoef(values,scores)[0,1]))
    (OUT/'sensitivity.json').write_text(json.dumps(sensitivity,indent=2));print('sensitivity',sensitivity)

if __name__=='__main__':
    result=fixtures();(OUT/'fixtures.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if '--fixtures-only' not in sys.argv:analyze()

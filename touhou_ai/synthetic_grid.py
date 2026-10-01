"""Exact specialized encoding of the restricted FixedDodge scene only."""
from functools import lru_cache
import numpy as np
from touhou_ai.dual_grid import paint_bullet_coverage,bin_entities
@lru_cache(maxsize=256)
def background(key):
    bullets=[{'position':[x,y],'velocity_raw':[0.,3.]} for x,y in key]
    grid=np.zeros((12,56,48),np.float32)
    for channel,frames in enumerate((0,2,4)):bin_entities(bullets,(channel,),grid,frames)
    return np.clip(grid,-1,1).astype(np.float16).astype(np.float32)
def encode_synthetic(raw):
    p=raw['player'];pos=np.array(p['position']);bullets=raw['bullets']
    # Future simulator variations must fall back rather than silently misencode.
    if (not bullets or raw['stage']!=1 or raw['lives_raw']!=0 or raw['power_raw']!=0
        or p['hitbox_raw']!=[1,1] or p['velocity_raw']!=[0,0] or p['invincibility_raw']!=0
        or any(raw[k] for k in ['enemies','items','lasers','player_shots'])
        or raw['bomb']!={'state':0} or raw['spell'] is not None
        or any(b['hitbox_raw']!=[8.,8.] or b['velocity_raw']!=[0.,3.] or b['flags_raw']!=2 for b in bullets)):
        from touhou_ai.dual_grid import DualGridContract
        return DualGridContract().encode(raw)
    key=tuple(tuple(b['position']) for b in bullets)
    whole=background(key).copy();px,py=pos+[192,0]
    if 0<=px<384 and 0<=py<448:whole[7,int(py//8),int(px//8)]=1
    local=np.zeros((6,96,96),np.float32);local[0,47:49,47:49]=.25
    positions=np.array([b['position'] for b in bullets]);sizes=np.full_like(positions,8.)
    for channel,frames in enumerate((0,2,4),1):paint_bullet_coverage(local[channel],positions+[0,frames*3],sizes,pos-96)
    player=np.array([pos[0]/192,pos[1]/448,0,0,0,0,p['status']/4,0,p['focus_raw'],1,1,1,1,1/6,1,1,0,0,0,False,0],np.float32)
    return {'local_grid':local.astype(np.float16).astype(np.float32),'global_grid':whole,'player':player,'previous_rewards':np.zeros(4,np.float32),'bomb_clock':np.zeros(1,np.float32)}

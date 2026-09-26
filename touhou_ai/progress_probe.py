"""Read-only ECL inventory and bounded diagnostic gameplay, no model updates."""
import argparse
import json
from pathlib import Path
import re
import struct
import math
from touhou_ai.live_runtime import LiveRuntime
from touhou_ai.live_reset import resume_paused_episode, continue_episode
from touhou_ai.live_acceptance import pause
from touhou_ai.live_transition import bridge_transition


def diagnostic_input(state, shoot):
    """Deterministic short-horizon dodge using observed bullet velocity, no training."""
    x,y=state['player']['position']
    bosses=[e for e in state['enemies'] or [] if e['is_boss']]
    tx=bosses[0]['position'][0] if bosses and shoot else 192
    bullets=[b for b in state['bullets'] or [] if (b['position'][0]-x)**2+(b['position'][1]-y)**2<180**2]
    choices=[]
    for dx,dy,mask in [(0,0,0),(1,0,0x80),(-1,0,0x40),(0,1,0x20),(0,-1,0x10),
                       (1,1,0xa0),(-1,1,0x60),(1,-1,0x90),(-1,-1,0x50)]:
        speed=2/math.sqrt(2) if dx and dy else 2
        risk=0
        for t in (2,6,12,20):
            px=max(4,min(380,x+dx*speed*t));py=max(4,min(444,y+dy*speed*t))
            for b in bullets:
                bx,by=b['position'];vx,vy=b['velocity_raw']
                distance=math.hypot(bx+vx*t-px,by+vy*t-py)
                risk+=1000*math.exp(-distance**2/200)/(1+t*.03)
        target=abs(x+dx*speed*8-tx)*.12+abs(y+dy*speed*8-390)*.2
        edge=max(0,24-(x+dx*speed*8))+max(0,x+dx*speed*8-360)
        choices.append((risk+target+edge,mask,risk))
    _,mask,risk=min(choices)
    mask|=4|(1 if shoot else 0)
    danger=any(math.hypot(b['position'][0]-x,b['position'][1]-y)<64 for b in bullets)
    if shoot and danger and state['player']['invincibility_raw']<=10 and state['power_raw']>=20 and state['bomb']['state']==0:
        mask|=2
    return mask


def inventory(reader):
    manager=reader.integer(0x477704); db=reader.integer(manager+0x54)
    count=reader.integer(db+8); table=reader.integer(db+0x8c)
    if not 0 < count <= 1024:
        raise ValueError('ECL subroutine count')
    entries=[]
    for i in range(count):
        name,header=struct.unpack('<II',reader.block(table+8*i,8))
        name=reader.block(name,96).split(b'\0')[0].decode('ascii')
        if reader.block(header,4)!=b'ECLH':
            raise ValueError('ECL subroutine magic')
        entries.append({'name':name,'header':header})
    entries.sort(key=lambda x:x['header'])
    for i,e in enumerate(entries):
        p=e['header']+16
        limit=min(e['header']+65536,entries[i+1]['header'] if i+1<len(entries) else e['header']+4096)
        e['instructions']=[]
        for _ in range(2048):
            if p+16>limit:break
            data=reader.block(p,16)
            time,op,size,flags,difficulty,argc=struct.unpack_from('<iHHHBB',data)
            if size<16 or size>2048 or p+size>limit:break
            args=reader.block(p+16,size-16) if size>16 else b''
            e['instructions'].append({'address':p,'time':time,'opcode':op,'size':size,
                'flags':flags,'difficulty':difficulty,'argc':argc,'args':args.hex(),
                'strings':[x.decode('ascii') for x in re.findall(rb'[A-Za-z_][A-Za-z_0-9]{3,}',args)]})
            p+=size
    return entries


def run(output,pid,frames,shoot,withhold_until=0):
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','training_updates':0,'frames_budget':frames,'shoot':shoot,'withhold_until':withhold_until}
    rt=LiveRuntime(pid,Path(__file__).with_suffix('.js'))
    try:
        (output/'ecl.json').write_text(json.dumps(inventory(rt.reader),indent=2),encoding='utf-8')
        state=rt.snapshot()
        if state['lives_raw']<0:continue_episode(rt)
        elif state['pause_words'][1]==2:resume_paused_episode(rt)
        with (output/'trace.jsonl').open('x',encoding='utf-8') as stream:
            for i in range(frames//2):
                before=rt.snapshot()
                mask=diagnostic_input(before,shoot and before['stage_frame']>=withhold_until)
                if before['dialogue_raw']:
                    mask=1 if i%2 else 0
                state=rt.step_gameplay(mask,2)
                trace=rt.api.progresstrace()
                if trace['error']:raise RuntimeError(trace['error'])
                stream.write(json.dumps({'state':state,'input':mask,'progress_probe':trace['events']})+'\n');stream.flush()
                if trace['events']:print(json.dumps({'frame':state['stage_frame'],'events':trace['events']}),flush=True)
                if state.get('transition')!='gameplay':
                    report['boundary']=state.get('transition');break
        report.update(status='completed',last_frame=state['stage_frame'])
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:
        try:
            current=rt.snapshot(full=False)
            if current['lives_raw']>=0 and (current['player'] is None or current['stage_frame']==0):
                current,report['cleanup_transition']=bridge_transition(rt,current,before['stage'])
            report['final']=pause(rt)
        except BaseException as error:
            report.update(status='cleanup_failed',cleanup_error=str(error))
            raise
        finally:
            rt.close();(output/'status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=1800)
    p.add_argument('--shoot',action='store_true');p.add_argument('--withhold-until',type=int,default=0);a=p.parse_args()
    if not 2<=a.frames<=18000:raise ValueError('bounded frame budget required')
    run(a.output,a.pid,a.frames,a.shoot,a.withhold_until)

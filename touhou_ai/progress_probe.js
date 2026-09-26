// Diagnostic only: no reward awards, stat writes, or synthetic game events.
let progressTrace = [], progressError = null;
function probeString(p) {
    if(p.isNull())return null;
    const bytes=Array.from(new Uint8Array(p.readByteArray(96))), end=bytes.indexOf(0);
    if(end<0 || bytes.slice(0,end).some(b=>b<32 || b>126))throw new Error('invalid ECL name');
    return String.fromCharCode(...bytes.slice(0,end));
}
function probeRecord(kind, owner, detail) {
    if (progressTrace.length >= 2048) throw new Error('progress trace overflow');
    progressTrace.push({kind, frame:ptr(0x474c88).readS32(), stage:ptr(0x474c7c).readS32(),
        owner:owner.toUInt32(), hp:owner.add(0x23fc).readS32(), flags:owner.add(0x2480).readU32(),
        lives_raw:ptr(0x474c70).readS32(), power_raw:ptr(0x474c48).readS32(), ...detail});
}
hooks.push(Interceptor.attach(ptr(0x4127a0), {
    onEnter() {
        this.owner=this.context.ecx;
        this.hp=this.owner.add(0x23fc).readS32();
        this.thresholds=[];
        for(let i=0;i<8;i++) {
            const p=this.owner.add(0x2494+i*16);
            this.thresholds.push({hp:p.readS32(),timer:p.add(4).readS32()});
        }
    },
    onLeave(value) {
        if(value.isNull())return;
        try {probeRecord('callback',this.owner,{name:probeString(value), hp_before:this.hp,thresholds:this.thresholds});}
        catch(e){progressError=String(e);}
    }
}));
hooks.push(Interceptor.attach(ptr(0x40e770), {
    onEnter() {
        try {
            const runtime=this.context.ecx, owner=runtime.add(0x14d8).readPointer();
            const instruction=owner.add(4).readPointer().add(4).readPointer();
            const opcode=instruction.add(4).readU16();
            if(![0x14b,0x14c,0x14e].includes(opcode))return;
            probeRecord('instruction',owner,{opcode,address:instruction.toUInt32(),
                operand_flags:instruction.add(8).readU16(),arg0:instruction.add(16).readS32(),
                bytes:Array.from(new Uint8Array(instruction.readByteArray(instruction.add(6).readU16())))});
        } catch(e){progressError=String(e);}
    }
}));
hooks.push(Interceptor.attach(ptr(0x450470), {
    onEnter(args) {
        try {
            const name=probeString(args[0]);
            if(name && /^(M?Boss|main|MainSub)/.test(name))
                progressTrace.push({kind:'subroutine',name,frame:ptr(0x474c88).readS32(),stage:ptr(0x474c7c).readS32()});
        }catch(e){progressError=String(e);}
    }
}));
rpc.exports.progresstrace=function(){const result={events:progressTrace,error:progressError};progressTrace=[];return result;};

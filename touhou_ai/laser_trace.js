// Bounded read-only arguments/results trace; never calls or changes the collision routine.
'use strict';
if (Process.arch !== 'ia32') throw new Error('unsupported architecture');
const entry = ptr(0x4267f0);
const bytes = Array.from(new Uint8Array(entry.readByteArray(9))).map(b=>b.toString(16).padStart(2,'0')).join('');
if (bytes !== '83ec24d981c0030000') throw new Error('laser collision signature mismatch');
let rows = [], overflow = false, fault = null;
const xy = p => [p.readFloat(), p.add(4).readFloat()];
const hook = Interceptor.attach(entry, {
    onEnter() {
        this.row = null;
        if (!this.returnAddress.equals(ptr(0x41d6c3))) return;
        try {
            const p = this.context.ecx, beam = this.context.ebp, stack = this.context.esp;
            if (!beam.readPointer().equals(ptr(0x46da60))) throw new Error('unexpected line vtable');
            const gui = ptr(0x47770c).readPointer();
            this.row = {frame:ptr(0x474c88).readS32(), laser:beam.toString(),
                position:xy(beam.add(0x24)), state:beam.add(0xc).readS32(),
                raw:[beam.add(0x3c).readFloat(),beam.add(0x40).readFloat(),beam.add(0x44).readFloat()],
                origin:xy(this.context.eax), angle:stack.add(4).readFloat(),
                width:stack.add(8).readFloat(), length:stack.add(12).readFloat(),
                player:xy(p.add(0x3c0)), halfbox:xy(p.add(0x41c)),
                player_status:p.add(0x458).readS32(), invincibility:p.add(0x4310).readS32(),
                dialogue:gui.isNull()?0:gui.add(0x9eb8).readU32()};
        } catch (error) { fault = String(error); }
    },
    onLeave(result) {
        if (this.row === null) return;
        this.row.result = result.toInt32();
        if (rows.length >= 100000) overflow = true;
        else rows.push(this.row);
    }
});
rpc.exports = {
    drain() { const batch=rows; rows=[]; return {rows:batch, overflow, fault}; },
    dispose() { hook.detach(); }
};

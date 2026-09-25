// Local in-process bridge; no file patches, game-stat writes, or OS key injection.
'use strict';
if (Process.arch !== 'ia32' || !Process.mainModule.base.equals(ptr(0x400000)))
    throw new Error('unsupported game process');
const kernel = Process.getModuleByName('kernel32.dll');
const createEvent = new NativeFunction(kernel.getExportByName('CreateEventW'), 'pointer', ['pointer', 'int', 'int', 'pointer']);
const setEvent = new NativeFunction(kernel.getExportByName('SetEvent'), 'int', ['pointer']);
const waitEvent = new NativeFunction(kernel.getExportByName('WaitForSingleObject'), 'uint', ['pointer', 'uint']);
const closeHandle = new NativeFunction(kernel.getExportByName('CloseHandle'), 'int', ['pointer']);
const asyncKey = new NativeFunction(Process.getModuleByName('user32.dll').getExportByName('GetAsyncKeyState'), 'int16', ['int']);
const wake = createEvent(ptr(0), 0, 0, ptr(0));
if (wake.isNull()) throw new Error('event allocation failed');
let enabled = false, parked = false, closed = false, tick = 0, target = 0;
let mask = 0, previous = 0, fault = null, calls = 0, handleClosed = false;
let lastCommand = Date.now();
function releaseHandle() { if (!handleClosed) { closeHandle(wake); handleClosed = true; } }
function clearInput() {
    mask = previous = 0;
    ptr(0x474e30).writeByteArray(new Uint8Array(0x2a));
}
function disable(reason) {
    enabled = false;
    fault = reason;
    clearInput();
    setEvent(wake);
}
function signature(address, hex) {
    const actual = Array.from(new Uint8Array(ptr(address).readByteArray(hex.length / 2))).map(b => b.toString(16).padStart(2,'0')).join('');
    if (actual !== hex) throw new Error('instruction signature mismatch at ' + address.toString(16));
}
signature(0x449c00, '83ec08538b1db0604600');
signature(0x44a5f0, '81ec08010000a160364700');
const hooks = [];
hooks.push(Interceptor.attach(ptr(0x44a5f0), {
    onEnter() { this.primary = this.context.ecx.toInt32() === 0; },
    onLeave() {
        if (!enabled || !this.primary) return;
        calls++;
        // Keep held, previous, repeat, pressed and released words coherent.
        const p = ptr(0x474e30);
        p.writeU16(mask); p.add(2).writeU16(previous); p.add(4).writeU16(0);
        p.add(6).writeU16(mask & ~previous); p.add(8).writeU16(previous & ~mask);
        previous = mask;
    }
}));
hooks.push(Interceptor.attach(ptr(0x449c00), {
    onEnter() {
        tick++;
        if (!enabled) return;
        if ((asyncKey(0x77) & 0x8000) !== 0) { disable('emergency F8'); return; }
        if (Date.now()-lastCommand > 3000) { disable('command lease expired'); return; }
        if (tick < target) return;
        parked = true;
        send({type:'parked', tick, frame:ptr(0x474c88).readS32()});
        const status = waitEvent(wake, 3000);
        parked = false;
        if (status !== 0) disable('frame gate timeout');
        if (closed) releaseHandle();
    }
}));
rpc.exports = {
    arm() {
        if (closed || enabled || fault !== null) throw new Error('gate unavailable');
        clearInput(); enabled = true; target = tick + 1; lastCommand = Date.now();
    },
    step(expected, frames, input) {
        if (!enabled || !parked || expected !== tick) throw new Error('stale/unparked command');
        if (!Number.isInteger(frames) || frames < 1 || frames > 120 || !Number.isInteger(input) || input < 0 || (input & ~0xff))
            throw new Error('invalid step');
        mask = input; target = tick + frames; lastCommand = Date.now(); setEvent(wake);
    },
    status() { return {enabled, parked, tick, target, mask, fault, input_calls:calls}; },
    stop() { disable('stopped'); },
    dispose() {
        if (closed) return;
        closed = true;
        disable('closed');
        for (const hook of hooks) hook.detach();
        if (!parked) releaseHandle();
    }
};

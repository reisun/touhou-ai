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
let gameplayGuard = false;
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
// Observe actual damage application, not HP differences between snapshots.
signature(0x40e1b6, '298dc0130000');
signature(0x40e231, '8b85bc130000');
signature(0x40da27, '899efc230000');
signature(0x40dae0, '518b0d04774700');
let combatEvents = [], combatError = null, combatSequence = 0, lifetimeSequence = 0;
const combatSession = Date.now().toString(36) + '-' + Process.id;
const enemyLifetimes = new Map();
function enemyIdentity(owner) {
    const key = owner.toString();
    if (!enemyLifetimes.has(key)) {
        if (enemyLifetimes.size >= 4096) throw new Error('enemy lifetime capacity exceeded');
        enemyLifetimes.set(key, {id: ++lifetimeSequence, lethalTick: -1, killed: false});
    }
    return enemyLifetimes.get(key);
}
function combatEvent(kind, owner, amount, evidence) {
    if (combatEvents.length >= 4096) throw new Error('combat event overflow');
    // Read at the actual event instruction, never at the end of a 2F action.
    const bomb = ptr(0x4776ec).readPointer();
    const bombState = bomb.isNull() ? null : bomb.add(0x28).readS32();
    const power = ptr(0x474c48).readS32();
    if ((bombState !== 0 && bombState !== 1) || power < 0 || power > 100)
        throw new Error('unknown event-time bomb/power state');
    combatEvents.push({id: combatSession + ':' + (++combatSequence), kind, amount,
        bomb_state: bombState, power_raw: power,
        confirmed: true, source: 'verified_game_event', enemy_id: enemyIdentity(owner).id,
        address: owner.toUInt32(), stage: ptr(0x474c7c).readS32(),
        frame: ptr(0x474c88).readS32(), gate_tick: tick, evidence});
}
hooks.push(Interceptor.attach(ptr(0x40da27), function () {
    enemyLifetimes.delete(this.context.esi.toString());
}));
hooks.push(Interceptor.attach(ptr(0x40dae0), function (args) {
    enemyLifetimes.delete(args[0].toString());
}));
hooks.push(Interceptor.attach(ptr(0x40e1b6), function () {
    if (!enabled || !gameplayGuard) return;
    try {
        const runtime = this.context.ebp, owner = runtime.add(0x14d8).readPointer();
        const hp = runtime.add(0x13c0).readS32(), damage = this.context.ecx.toInt32();
        if (!runtime.sub(0x103c).equals(owner) || damage < 0 || damage > 10000000)
            throw new Error('unexpected enemy damage layout/value');
        let floor = 0;
        for (let i = 0; i < 8; i++) {
            const threshold = owner.add(0x2494 + i * 16).readS32();
            if (threshold >= 0) { floor = threshold; break; }
        }
        const effective = Math.min(damage, Math.max(0, hp - floor));
        const identity = enemyIdentity(owner);
        identity.lethalTick = hp > 0 && damage >= hp && floor === 0 ? tick : -1;
        if (effective > 0) combatEvent('damage', owner, effective,
            {instruction: '40e1b6', hp_before: hp, applied_damage: damage, hp_floor: floor,
             boss: !!(runtime.add(0x1444).readU32() & 0x8000)});
    } catch (error) { combatError = String(error); }
}));
hooks.push(Interceptor.attach(ptr(0x40e231), function () {
    if (!enabled || !gameplayGuard) return;
    try {
        const runtime = this.context.ebp, owner = runtime.add(0x14d8).readPointer();
        const identity = enemyIdentity(owner), hp = runtime.add(0x13c0).readS32();
        if (hp <= 0 && identity.lethalTick === tick && !identity.killed) {
            identity.killed = true;
            combatEvent('kill', owner, 1, {instruction: '40e231', hp_after: hp});
        }
    } catch (error) { combatError = String(error); }
}));
hooks.push(Interceptor.attach(ptr(0x44a5f0), {
    onEnter() { this.primary = this.context.ecx.toInt32() === 0; },
    onLeave() {
        if (!enabled || !this.primary) return;
        calls++;
        // A policy's held shot must never become a menu confirmation after death.
        if (gameplayGuard && ptr(0x474c70).readS32() < 0) clearInput();
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
        if (gameplayGuard && ptr(0x474c70).readS32() < 0) clearInput();
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
    step(expected, frames, input, guard) {
        if (!enabled || !parked || expected !== tick) throw new Error('stale/unparked command');
        if (!Number.isInteger(frames) || frames < 1 || frames > 120 || !Number.isInteger(input) || input < 0 || (input & ~0xff))
            throw new Error('invalid step');
        gameplayGuard = guard === true;
        combatEvents = [];
        mask = input; target = tick + frames; lastCommand = Date.now(); setEvent(wake);
    },
    status() { return {enabled, parked, tick, target, mask, fault, input_calls:calls}; },
    combat() { return {events: combatEvents, error: combatError}; },
    stop() { disable('stopped'); },
    dispose() {
        if (closed) return;
        closed = true;
        disable('closed');
        for (const hook of hooks) hook.detach();
        if (!parked) releaseHandle();
    }
};

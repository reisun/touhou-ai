// Pinned TH10 + th10.dat. Engine paths verified live; stage mappings reviewed in ECL.
signature(0x4127a0, '535533ed5633c08d9194240000');
signature(0x40e770, '558bec83e4f881ecc4020000');
let progressEvents = [], progressFailure = null;
function eclName(p) {
    if (p.isNull()) return null;
    const bytes = Array.from(new Uint8Array(p.readByteArray(96))), end = bytes.indexOf(0);
    if (end < 0 || bytes.slice(0, end).some(b => b < 32 || b > 126))
        throw new Error('invalid ECL name');
    return String.fromCharCode(...bytes.slice(0, end));
}
function progressContext() {
    const replay = ptr(0x477838).readPointer(), stage = ptr(0x474c7c).readS32();
    return enabled && stage >= 1 && stage <= 6 && ptr(0x474c74).readS32() === 1
        && ptr(0x474c68).readS32() === 0 && ptr(0x474c6c).readS32() === 1
        && [0, 4].includes(ptr(0x474ca0).readS32()) && !replay.isNull() && replay.add(0x10).readS32() === 0;
}
function instructionSub(owner, instruction) {
    const db = owner.add(0x102c).readPointer(), count = db.add(8).readU32();
    if (count < 1 || count > 1024) throw new Error('invalid ECL table count');
    const table = db.add(0x8c).readPointer();
    let nearest = null, distance = 65536;
    for (let i = 0; i < count; i++) {
        const entry = table.add(i * 8), header = entry.add(4).readPointer();
        const delta = instruction.toUInt32() - header.toUInt32();
        if (delta >= 16 && delta < distance) { nearest = entry; distance = delta; }
    }
    if (nearest === null) throw new Error('instruction outside ECL table');
    if (nearest.add(4).readPointer().readU32() !== 0x484c4345) throw new Error('invalid ECLH');
    return eclName(nearest.readPointer());
}
function addProgress(owner, milestone, evidence, spellId=null) {
    if (progressEvents.length >= 64) throw new Error('progress event overflow');
    const lives = ptr(0x474c70).readS32(), power = ptr(0x474c48).readS32();
    if (lives < 0) return; // No progress bonus after game over.
    if (lives > 8 || power < 0 || power > 100) throw new Error('invalid progress event state');
    const bomb = ptr(0x4776ec).readPointer();
    const bombState = bomb.isNull() ? null : bomb.add(0x28).readS32();
    if (bombState !== 0 && bombState !== 1) throw new Error('unknown event-time bomb state');
    progressEvents.push({id: combatSession + ':p:' + (++combatSequence), kind: 'progress',
        confirmed: true, source: 'verified_ecl_progress_v2', milestone,
        difficulty_raw: 1, spell_id_raw: spellId,
        stage: ptr(0x474c7c).readS32(), frame: ptr(0x474c88).readS32(), gate_tick: tick,
        enemy_id: enemyIdentity(owner).id, lives_raw: lives, power_raw: power, bomb_state: bombState, evidence});
}
hooks.push(Interceptor.attach(ptr(0x40e770), {
    onEnter() {
        try {
            if (!enabled) return;
            const runtime = this.context.ecx, owner = runtime.add(0x14d8).readPointer();
            const instruction = owner.add(4).readPointer().add(4).readPointer();
            if (instruction.add(4).readU16() !== 0x14c) return;
            if (!progressContext()) return;
            if (instruction.add(6).readU16() !== 20 || instruction.add(8).readU16() !== 0)
                throw new Error('unexpected boss registration operands');
            if (instruction.add(16).readS32() !== 0) return;
            const stage = ptr(0x474c7c).readS32(), name = instructionSub(owner, instruction);
            const role = name === 'Boss' ? 'boss' : stage <= 5 &&
                (name === 'MBoss' || (stage === 3 && name === 'MBossDummy')) ? 'midboss' : null;
            if (role === null) return;
            const identity = enemyIdentity(owner);
            identity.progressRole = role; identity.progressStage = stage;
            addProgress(owner, role + '_arrival', {instruction: '40e770', subroutine: name, opcode: 332});
        } catch (error) { progressFailure = String(error); }
    }
}));
hooks.push(Interceptor.attach(ptr(0x4127a0), {
    onEnter() {
        this.observe = false;
        try {
            if (!enabled) return;
            this.owner = this.context.ecx;
            const identity = enemyLifetimes.get(this.owner.toString());
            if (!identity || !identity.progressRole || !progressContext()
                    || identity.progressStage !== ptr(0x474c7c).readS32()) return;
            this.role = identity.progressRole; this.stage = identity.progressStage;
            this.identity = identity;
            const spell = ptr(0x4776f4).readPointer();
            this.spell = null;
            if (!spell.isNull() && (spell.add(0x378c).readU32() & 1)) {
                const id = spell.add(0x3788).readS32();
                this.spell = NORMAL_SPELL_PROGRESS.find(s => s.id === id && s.stage === this.stage && s.role === this.role) || null;
            }
            this.hp = this.owner.add(0x23fc).readS32(); this.observe = true;
        } catch (error) { progressFailure = String(error); }
    },
    onLeave(value) {
        if (!this.observe || value.isNull()) return;
        try {
            if (!progressContext() || this.stage !== ptr(0x474c7c).readS32()) return;
            const flags = this.owner.add(0x2480).readU32(), hp = this.owner.add(0x23fc).readS32();
            const name = eclName(value);
            const timeout = !!(flags & 0x10000), spell = this.spell;
            if (spell && name === spell.callback && hp === spell.hp && (timeout || this.hp <= spell.hp)) {
                const completed = this.identity.completedSpells || (this.identity.completedSpells = new Set());
                if (completed.has(spell.id)) return;
                addProgress(this.owner, spell.final ? this.role + '_defeat' : 'spell_breakthrough',
                    {instruction: '4127a0', callback: name, flags, hp_before: this.hp, hp_after: hp,
                     timeout, spell_id_raw: spell.id, final_spell: spell.final}, spell.id);
                completed.add(spell.id);
                return;
            }
            const final = this.role === 'boss' ? 'BossDead' : this.stage === 5 ? 'MBossEscape' : 'MBossDead';
            if (name !== final || (flags & 0x10000) || this.hp > 0 || hp !== 0) return;
            addProgress(this.owner, this.role + '_defeat',
                {instruction: '4127a0', callback: name, flags, hp_before: this.hp, hp_after: hp, timeout: false});
        } catch (error) { progressFailure = String(error); }
    }
}));
const stepWithoutProgress = rpc.exports.step;
rpc.exports.step = function (...args) { progressEvents = []; return stepWithoutProgress(...args); };
rpc.exports.progress = function () { return {events: progressEvents, error: progressFailure}; };

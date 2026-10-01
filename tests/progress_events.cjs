const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const memory = new Map(), strings = new Map(), callbacks = new Map(), identities = new Map();
function ptr(address) { return {
  add:n=>ptr(address+n), toUInt32:()=>address, toString:()=>String(address), isNull:()=>address===0,
  readPointer:()=>ptr(memory.get(address)||0), readS32:()=>memory.get(address)||0,
  readU32:()=>memory.get(address)||0, readU16:()=>memory.get(address)||0,
  readByteArray:n=>Uint8Array.from([...Buffer.from(strings.get(address)||''), ...Array(n).fill(0)].slice(0,n)).buffer
}; }
const context = {ptr, Uint8Array, signature(){}, hooks:[], enabled:true, tick:1,
  combatSession:'fixture', combatSequence:0, enemyLifetimes:identities,
  enemyIdentity:p=>{const key=p.toString();if(!identities.has(key))identities.set(key,{id:identities.size+1});return identities.get(key);},
  Interceptor:{attach:(p,h)=>{callbacks.set(p.toUInt32(),h);return {detach(){}};}}, rpc:{exports:{step(){}}}};
vm.createContext(context);
context.NORMAL_SPELL_PROGRESS=JSON.parse(fs.readFileSync('touhou_ai/spell_progress.json','utf8'));
vm.runInContext(fs.readFileSync('touhou_ai/progress_events.js','utf8'),context);
const api=context.rpc.exports, register=callbacks.get(0x40e770), finish=callbacks.get(0x4127a0);
const owner=0x100000, runtime=owner+0x103c, ctx=0x200000, db=0x210000, table=0x220000, header=0x230000, instruction=header+32;
for(const [a,v] of [[0x477838,0x240000],[0x474c74,1],[0x474c6c,1],[0x474c70,2],[0x474c48,40],
 [0x4776ec,0x280000],
 [runtime+0x14d8,owner],[owner+4,ctx],[ctx+4,instruction],[owner+0x102c,db],[db+8,1],[db+0x8c,table],
 [table,0x250000],[table+4,header],[header,0x484c4345],[instruction+4,332],[instruction+6,20]])memory.set(a,v);
function arrival(stage,name){api.step();identities.clear();memory.set(0x474c7c,stage);strings.set(0x250000,name);register.onEnter.call({context:{ecx:ptr(runtime)}});}
function callback(name,timeout=false,hp=0){memory.set(owner+0x23fc,hp);const call={context:{ecx:ptr(owner)}};finish.onEnter.call(call);memory.set(owner+0x23fc,0);memory.set(owner+0x2480,timeout?0x10000:0);strings.set(0x260000,name);finish.onLeave.call(call,ptr(0x260000));}
for(let stage=1;stage<=6;stage++) {
  arrival(stage,'Boss');assert.equal(api.progress().events[0].milestone,'boss_arrival');
  callback('Boss2');assert.equal(api.progress().events.length,1); // Intermediate zero HP.
  callback('BossDead',true,8000);assert.equal(api.progress().events.length,1); // Timeout sets HP zero too.
  callback('BossDead',false,-4);assert.equal(api.progress().events[1].milestone,'boss_defeat');
  arrival(stage,'MBoss');
  if(stage===6){assert.equal(api.progress().events.length,0);continue;}
  assert.equal(api.progress().events[0].milestone,'midboss_arrival');
  callback('MBossEscape',true,4000);assert.equal(api.progress().events.length,1);
  callback(stage===5?'MBossEscape':'MBossDead');assert.equal(api.progress().events[1].milestone,'midboss_defeat');
}
arrival(3,'MBossDummy');assert.equal(api.progress().events[0].milestone,'midboss_arrival');
memory.set(0x474ca0,4);
arrival(3,'MBossDummy');assert.equal(api.progress().events[0].milestone,'midboss_arrival');
for(const flags of [5,6,12,36]){
 memory.set(0x474ca0,flags);arrival(3,'MBossDummy');assert.equal(api.progress().events.length,0);
}
memory.set(0x474ca0,0);
arrival(3,'MBossDummy');
memory.set(0x474c70,0);memory.set(0x474c48,100);
assert.equal(api.progress().events[0].lives_raw,2);assert.equal(api.progress().events[0].power_raw,40);
api.step();identities.clear();callback('MBossDead');assert.equal(api.progress().events.length,0);
arrival(2,'BossDummyShadow');assert.equal(api.progress().events.length,0);
memory.set(0x240010,1);arrival(1,'Boss');assert.equal(api.progress().events.length,0);memory.set(0x240010,0);
// Every reviewed spell supports HP and timeout completion; final spells fold into boss end.
const spellBase=0x290000;memory.set(0x4776f4,spellBase);
for(const s of context.NORMAL_SPELL_PROGRESS)for(const timeout of [false,true]){
 memory.set(spellBase+0x378c,0);arrival(s.stage,s.role==='boss'?'Boss':'MBoss');api.step();
 memory.set(spellBase+0x378c,1);memory.set(spellBase+0x3788,s.id);
 memory.set(owner+0x23fc,timeout?s.hp+100:s.hp-1);
 const call={context:{ecx:ptr(owner)}};finish.onEnter.call(call);
 memory.set(owner+0x23fc,s.hp);memory.set(owner+0x2480,timeout?0x10000:0);strings.set(0x260000,s.callback);
 finish.onLeave.call(call,ptr(0x260000));
 const events=api.progress().events;assert.equal(events.length,1,JSON.stringify(s));
 assert.equal(events[0].milestone,s.final?s.role+'_defeat':'spell_breakthrough');
 assert.equal(events[0].spell_id_raw,s.id);assert.equal(events[0].evidence.timeout,timeout);
 finish.onLeave.call(call,ptr(0x260000));assert.equal(api.progress().events.length,1);
}
// No completion from stale inactive IDs, game over, retry frame loss, or other difficulty.
memory.set(spellBase+0x378c,0);arrival(1,'Boss');api.step();memory.set(spellBase+0x3788,3);
callback('Boss2');assert.equal(api.progress().events.length,0);
for(const [address,value] of [[0x474c70,-1],[0x474ca0,1],[0x474c74,2]]){
 memory.set(spellBase+0x378c,0);arrival(1,'Boss');api.step();memory.set(spellBase+0x378c,1);
 const old=memory.get(address)||0;memory.set(address,value);callback('Boss2');
 assert.equal(api.progress().events.length,0);memory.set(address,old);
}
memory.set(spellBase+0x378c,0);
memory.set(instruction+8,1);arrival(1,'Boss');assert.match(api.progress().error,/operands/);
console.log('Progress hooks: stage mappings, event-time state, zero-HP phases, timeouts, replay, owner lifetime and malformed operands passed.');

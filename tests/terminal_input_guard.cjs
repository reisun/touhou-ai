const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let lives = 2, held = 0;
const memory = new Map();
const hooks = new Map();
const signatures = {0x449c00:'83ec08538b1db0604600',0x44a5f0:'81ec08010000a160364700',
 0x40e1b6:'298dc0130000',0x40e231:'8b85bc130000',0x40da27:'899efc230000',
 0x40dae0:'518b0d04774700'};
function ptr(address) {return {
 equals: p => address === p.address, address,
 isNull: () => address === 0, readS32: () => address === 0x474c70 ? lives : (memory.get(address) ?? 0),
 readU32: () => memory.get(address) ?? 0, readPointer: () => ptr(memory.get(address) ?? 0),
 toString: () => String(address), toInt32: () => address, toUInt32: () => address,
 sub: n => ptr(address-n),
 readByteArray: () => Uint8Array.from(Buffer.from(signatures[address], 'hex')).buffer,
 writeByteArray: () => {if(address === 0x474e30) held = 0;},
 writeU16: v => {if(address === 0x474e30) held = v;}, add: n => ptr(address+n)
};}
const context = {ptr, Uint8Array, Date, rpc:{exports:{}}, send:()=>{},
 Process:{arch:'ia32',mainModule:{base:ptr(0x400000)},getModuleByName:()=>({getExportByName:n=>n})},
 NativeFunction:function(name){return (...args)=>name==='CreateEventW'?ptr(1):0;},
 Interceptor:{attach:(address, handlers)=>{hooks.set(address.address,handlers);return {detach(){}};}}
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('touhou_ai/th10_gate.js','utf8'),context);
const api=context.rpc.exports, input=hooks.get(0x44a5f0), gate=hooks.get(0x449c00);
api.arm();
function command(mask, guard){vm.runInContext('parked = true',context);api.step(api.status().tick,2,mask,guard);}
command(0x91,true);input.onLeave.call({primary:true});assert.equal(held,0x91);
lives=-1;input.onLeave.call({primary:true});assert.equal(held,0);
command(1,true);gate.onEnter();assert.equal(held,0);assert.equal(api.status().mask,0);
command(1,false);input.onLeave.call({primary:true});assert.equal(held,1);
console.log('Terminal guard clears policy input; explicit menu confirmation remains available.');
const owner=0x100000, runtime=owner+0x103c;
memory.set(0x4776ec,0x200000);
memory.set(0x200028,0);
memory.set(0x474c48,40);
memory.set(runtime+0x14d8,owner);
for(let i=0;i<8;i++) memory.set(owner+0x2494+i*16,-1);
const damage=hooks.get(0x40e1b6),kill=hooks.get(0x40e231),spawn=hooks.get(0x40da27);
function hit(hp,amount){memory.set(runtime+0x13c0,hp);damage.call({context:{ebp:ptr(runtime),ecx:ptr(amount)}});}
function death(){memory.set(runtime+0x13c0,0);kill.call({context:{ebp:ptr(runtime)}});}
command(1,true);hit(100,30);assert.equal(api.combat().events[0].amount,30);
command(1,true);hit(10,30);death();death();
assert.deepEqual(Array.from(api.combat().events,e=>[e.kind,e.amount]),[['damage',10],['kill',1]]);
command(1,true);death();assert.equal(api.combat().events.length,0);
spawn.call({context:{esi:ptr(owner)}});command(1,true);hit(10,30);death();
assert.equal(api.combat().events.filter(e=>e.kind==='kill').length,1);
memory.set(owner+0x2494,50);command(1,true);hit(60,30);
assert.equal(api.combat().events[0].amount,10);
command(0,false);hit(100,20);assert.equal(api.combat().events.length,0);
command(0,true);assert.equal(api.combat().events.length,0);
assert.equal(api.combat().error,null);
console.log('Combat hooks cap overkill/phase floors, deduplicate kills, reset reused lifetimes and exclude menu steps.');
memory.set(owner+0x2494,-1);
command(2,true);memory.set(0x200028,1);memory.set(0x474c48,20);hit(100,5);
memory.set(0x200028,0);memory.set(0x474c48,80);hit(95,5);
const captured=api.combat().events;
assert.deepEqual(Array.from(captured,e=>[e.bomb_state,e.power_raw]),[[1,20],[0,80]]);
memory.set(0x200028,1);memory.set(0x474c48,100);
assert.deepEqual(Array.from(captured,e=>[e.bomb_state,e.power_raw]),[[1,20],[0,80]]);
command(0,true);memory.set(0x4776ec,0);hit(100,5);
assert.match(api.combat().error,/unknown event-time/);
assert.equal(api.combat().events.length,0);
console.log('Event-time bomb/Power survives both frame boundaries; unknown bomb pointer fails closed.');

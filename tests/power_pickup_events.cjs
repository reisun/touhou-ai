const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
let callbacks, power=0, item=1;
function ptr(n){return {toUInt32:()=>n,toInt32:()=>n,equals:p=>p.toUInt32()===n,
 add:x=>ptr(n+x),readS16:()=>{assert.equal(n,0x474c48);return power;},
 readS32:()=>n===0x1030?item:n===0x474c7c?1:100};}
const c={ptr,enabled:true,gameplayGuard:true,hooks:[],combatEvents:[],combatError:null,
 combatSession:'test',combatSequence:0,tick:9,signature:()=>{},
 Interceptor:{attach:(p,cb)=>{assert.equal(p.toUInt32(),0x418930);callbacks=cb;return {};}}};
vm.runInNewContext(fs.readFileSync('touhou_ai/power_pickup_events.js','utf8'),c);
function pickup(type,before,after,caller){
 item=type;power=before;
 const amount=({1:1,10:1,4:20,11:20})[type];
 const self={context:{ebp:ptr(0x1000),eax:ptr(0x474c40)},returnAddress:ptr(caller??(amount===1?0x41b3e0:0x41b63a))};
 callbacks.onEnter.call(self,[ptr(amount)]);power=after;callbacks.onLeave.call(self);
}
pickup(1,0,1);pickup(10,1,2);pickup(4,2,22);pickup(11,95,100);pickup(4,100,100);
assert.deepEqual(Array.from(c.combatEvents,e=>e.amount_raw),[1,1,20,5]);
assert.equal(new Set(c.combatEvents.map(e=>e.id)).size,4);
assert.equal(c.combatError,null);
pickup(1,20,21,0x123456);assert.equal(c.combatEvents.length,4);
c.gameplayGuard=false;pickup(1,20,21);assert.equal(c.combatEvents.length,4);
c.gameplayGuard=true;pickup(4,20,21);assert.match(c.combatError,/unexpected actual Power gain/);
assert.equal(c.combatEvents.length,4);
console.log('Power pickup hook: aliases, cap, duplicate identity, caller/collection guards and invalid gain passed');

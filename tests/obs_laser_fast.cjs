const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const reference=require('./laser_coverage_reference.cjs');
const src=fs.readFileSync('dashboard/obs.js','utf8');
// Run both benchmark implementations in the same JS realm (including Math).
const ctx=new Function(src.slice(src.indexOf('function prepareLasers('),src.indexOf('function powerItemGrid('))+';return {grid:gridLaserCoverage,prepare:prepareLasers};')();
ctx.reference=reference;
vm.runInNewContext('const compact=true;function field(){throw Error("legacy field called");}'+src.slice(src.indexOf('const focusedField = field;'),src.indexOf('function node('))+';field({});');
let seed=813;const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
function verify(entities,center){
 const prepared=ctx.prepare(entities),hidden=[center[0]-96,center[1]-96,center[0]+96,center[1]+96];
 for(const [origin,cell,w,h,clip] of [[[0,0],8,48,56,hidden],[[center[0]-96,center[1]-96],2,96,96,null]]){
  const old=reference(entities,origin,cell,w,h),full=ctx.grid(entities,origin,cell,w,h,prepared),fast=ctx.grid(entities,origin,cell,w,h,prepared,clip);
  assert.deepEqual(Array.from(full),Array.from(old));
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const omitted=clip&&origin[0]+x*cell>=clip[0]&&origin[0]+(x+1)*cell<=clip[2]&&origin[1]+y*cell>=clip[1]&&origin[1]+(y+1)*cell<=clip[3];
   assert.equal(fast[y*w+x],omitted?0:old[y*w+x]);
  }
 }
}
for(let n=0;n<80;n++)verify(Array.from({length:12},(_,i)=>({collision:{active:i%7!==0,origin:[random()*600-300,random()*600-100],angle:n<8?n*Math.PI/4:random()*7,length:random()*650,width:i===1?0:random()*35}})),[random()*384,random()*448]);
const path='artifacts/fps-diagnosis-20260930/raws.json';
if(fs.existsSync(path)){
 const rows=JSON.parse(fs.readFileSync(path));const active=[];
 for(const {raw} of rows){const center=[raw.player.position[0]+192,raw.player.position[1]];verify(raw.lasers,center);if(raw.lasers.some(l=>l.collision.active))active.push({entities:raw.lasers,center});}
 function run(fast){const start=performance.now();for(const {entities,center} of active){const p=fast?ctx.prepare(entities):null;const g=fast?ctx.grid:ctx.reference;g(entities,[0,0],8,48,56,p,fast?[center[0]-96,center[1]-96,center[0]+96,center[1]+96]:null);g(entities,[center[0]-96,center[1]-96],2,96,96,p);}return (performance.now()-start)/active.length;}
 for(let i=0;i<4;i++){run(false);run(true);}
 const old=[],fast=[];for(let i=0;i<11;i++){if(i%2){fast.push(run(true));old.push(run(false));}else{old.push(run(false));fast.push(run(true));}}
 const median=a=>a.sort((a,b)=>a-b)[5];console.log(JSON.stringify({captured:rows.length,active:active.length,old_ms:median(old),new_ms:median(fast)}));
}
console.log('Laser grid parity, clipped cells, fractional viewports, and hidden obs2 passed');

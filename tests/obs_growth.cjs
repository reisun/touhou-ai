const fs=require('fs'),vm=require('vm'),assert=require('assert');
const src=fs.readFileSync(require('path').join(__dirname,'../dashboard/obs.js'),'utf8');
const fn=src.slice(src.indexOf('function drawGrowth('),src.indexOf('function renderStats('));
const calls=[];const context={fillRect:(...v)=>calls.push(['bar',...v]),fillText:(...v)=>calls.push(['text',...v]),beginPath(){},moveTo:(...v)=>calls.push(['move',...v]),lineTo:(...v)=>calls.push(['line',...v]),stroke(){},arc:(...v)=>calls.push(['dot',...v]),fill(){},clearRect(){}};
vm.runInNewContext(fn+';globalThis.draw=drawGrowth;',globalThis);
globalThis.fmt=(v,n)=>v.toFixed(n);
const el={width:498,height:230,getContext:()=>context};
for(const data of [[],[{return:3}],Array.from({length:24},(_,i)=>({return:i-10,max_progress:{rank:i+1}})),[{return:5,max_progress:{rank:4}},{return:-5},{return:1,max_progress:{rank:1}}]]){
 calls.length=0;draw(el,data);
 assert(calls.every(a=>a.slice(1).filter(v=>typeof v==='number').every(Number.isFinite)));
 assert(calls.filter(a=>a[0]==='dot').every(a=>a[1]>=44&&a[1]<=404&&a[2]>=22&&a[2]<=212));
 assert(calls.some(a=>a[0]==='text'&&a[1]==='スタート'));
 assert.equal(calls.filter(a=>a[0]==='dot').length,data.length);
 const bars=calls.filter(a=>a[0]==='bar');assert.equal(bars.length,data.length);
 assert(bars.every(a=>a[1]>=44&&a[1]+a[3]<=404&&a[2]>=22&&a[2]+a[4]===212));
 if(data.length===1)assert.equal(bars[0][4],2);
 if(data.length)assert(calls.findIndex(a=>a[0]==='bar')<calls.findIndex(a=>a[0]==='dot'));
 if(data.length===3)assert.equal(calls.filter(a=>a[0]==='line').length,5); // three grid + two mean; progress is bars and reward is dots
}
console.log('Growth drawing: empty, single, 6 stages, missing and decreasing points passed');
calls.length=0;
draw(el,[{return:10,max_progress:{rank:3.5}}],[
 {rank:1,label:'1面 中ボス着'},{rank:2,label:'1面 中ボス終了'},
 {rank:3,label:'1面 ボス着'},{rank:3.5,label:'1面 スペル突破'},{rank:4,label:'1面 ボス終了'}]);
assert(calls.some(a=>a[0]==='text'&&a[1]==='1面 スペル突破'));
assert.equal(calls.filter(a=>a[0]==='bar').length,1);
assert(calls.find(a=>a[0]==='bar')[4]>100);
console.log('Growth drawing: fractional spell rank and generic spell label passed');


const milestoneLabels=['スタート','1面 中ボス着','1面 中ボス突破','1面 ボス着','1面 スペル突破','1面 ボス突破'];
const ys=milestoneLabels.map(label=>calls.find(a=>a[0]==='text'&&a[1]===label)[3]);
for(let i=2;i<ys.length;i++)assert(Math.abs((ys[i-1]-ys[i])-(ys[0]-ys[1]))<1e-9);
assert(Math.abs(calls.find(a=>a[0]==='bar')[2]-(ys[4]-4))<1e-9);
console.log('Growth axis: spell and boss milestones equally spaced, bar aligned');
calls.length=0;
draw(el,[{return:3,max_progress:{rank:8}}]);
for(const stage of [1,2]){
 assert(calls.some(a=>a[0]==='text'&&a[1]===`${stage}面 中ボス突破`));
 assert(calls.some(a=>a[0]==='text'&&a[1]===`${stage}面 ボス突破`));
}
assert(!calls.some(a=>a[0]==='text'&&a[1].includes('ボス終了')));

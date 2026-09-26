const fs=require('fs'),vm=require('vm'),assert=require('assert');
const src=fs.readFileSync(require('path').join(__dirname,'../dashboard/obs.js'),'utf8');
const fn=src.slice(src.indexOf('function drawGrowth('),src.indexOf('function renderStats('));
const calls=[];const context={fillText:(...v)=>calls.push(['text',...v]),beginPath(){},moveTo:(...v)=>calls.push(['move',...v]),lineTo:(...v)=>calls.push(['line',...v]),stroke(){},arc:(...v)=>calls.push(['dot',...v]),fill(){},clearRect(){}};
vm.runInNewContext(fn+';globalThis.draw=drawGrowth;',globalThis);
globalThis.fmt=(v,n)=>v.toFixed(n);
const el={width:498,height:230,getContext:()=>context};
for(const data of [[],[{return:3}],Array.from({length:24},(_,i)=>({return:i-10,max_progress:{rank:i+1}})),[{return:5,max_progress:{rank:4}},{return:-5},{return:1,max_progress:{rank:1}}]]){
 calls.length=0;draw(el,data);
 assert(calls.every(a=>a.slice(1).filter(v=>typeof v==='number').every(Number.isFinite)));
 assert(calls.filter(a=>a[0]==='dot').every(a=>a[1]>=44&&a[1]<=404&&a[2]>=22&&a[2]<=212));
 assert(calls.some(a=>a[0]==='text'&&a[1]==='スタート'));
 assert.equal(calls.filter(a=>a[0]==='dot').length,data.length*2);
 if(data.length===1)assert(calls.some(a=>a[0]==='dot'&&a[2]===212));
 if(data.length===3)assert.equal(calls.filter(a=>a[0]==='line').length,7); // three grid + two mean + two progress; reward is dots only
}
console.log('Growth drawing: empty, single, 6 stages, missing and decreasing points passed');


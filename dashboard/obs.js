'use strict';
setTimeout(() => location.reload(), 5 * 60 * 1000);
const $=id=>document.getElementById(id), compact=new URLSearchParams(location.search).get('mode')==='obs2';
document.body.classList.toggle('compact',compact);$(compact?'obs2':'obs1').hidden=false;
const labels={progress:'進行度',damage:'ショットダメージ',power_gain:'Power取得',hit:'被弾',jitter:'振動減点'};
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';
const order=[8,1,2,7,0,3,6,5,4], icons=['circle','arrow-up','arrow-up-right','arrow-right','arrow-down-right','arrow-down','arrow-down-left','arrow-left','arrow-up-left'];
let frame=null, pending=null, stats=null;
// Display reconstruction from telemetry; this does not change model observations.
function localBulletCoverage(entities, center, frames=0) {
 const coverage=new Float32Array(96*96),ox=center[0]-96,oy=center[1]-96;
 for(const e of entities||[]) {
  if(Number.isInteger(e.flags_raw)&&!(e.flags_raw&2))continue;
  const size=e.hitbox_raw;
  if(!Array.isArray(size)||size.length!==2||!size.every(v=>Number.isFinite(v)&&v>0))continue;
  const v=e.velocity_raw;if(frames&&(!Array.isArray(v)||v.length!==2||!v.every(Number.isFinite)))continue;
  const x=e.position[0]+frames*(v?.[0]??0),y=e.position[1]+frames*(v?.[1]??0);
  const left=(x-size[0]/2-ox)/2,right=(x+size[0]/2-ox)/2;
  const top=(y-size[1]/2-oy)/2,bottom=(y+size[1]/2-oy)/2;
  for(let j=Math.max(0,Math.floor(top));j<Math.min(96,Math.ceil(bottom));j++)
   for(let i=Math.max(0,Math.floor(left));i<Math.min(96,Math.ceil(right));i++) {
    const area=Math.max(0,Math.min(i+1,right)-Math.max(i,left))*Math.max(0,Math.min(j+1,bottom)-Math.max(j,top));
    coverage[j*96+i]=Math.max(coverage[j*96+i],area);
   }
 }
 return coverage;
}
function prepareLasers(entities) {
 const result=[];
 for(const e of entities||[]) {
  const g=e.collision;if(!g?.active)continue;
  const px=g.origin[0]+192,py=g.origin[1],co=Math.cos(g.angle),si=Math.sin(g.angle);
  const corners=[];for(const t of [0,g.length])for(const w of [-g.width/2,g.width/2])corners.push([px+co*t-si*w,py+si*t+co*w]);
  result.push({px,py,co,si,length:g.length,half:g.width/2,
   left:Math.min(...corners.map(p=>p[0])),right:Math.max(...corners.map(p=>p[0])),
   top:Math.min(...corners.map(p=>p[1])),bottom:Math.max(...corners.map(p=>p[1]))});
 }
 return result;
}
function gridLaserCoverage(entities, origin, cell, width, height, prepared=null, hidden=null) {
 const coverage=new Float32Array(width*height);
 for(const g of prepared??prepareLasers(entities)) {
  const {px,py,co,si}=g;
  const x0=Math.max(0,Math.floor((g.left-origin[0])/cell)),x1=Math.min(width,Math.ceil((g.right-origin[0])/cell));
  const y0=Math.max(0,Math.floor((g.top-origin[1])/cell)),y1=Math.min(height,Math.ceil((g.bottom-origin[1])/cell));
  if(x0>=x1||y0>=y1)continue;
  const xc=new Float64Array((x1-x0)*2),xs=new Float64Array(xc.length);
  for(let x=x0;x<x1;x++)for(let s=0;s<2;s++) {
   const dx=origin[0]+(x+(s===0?.25:.75))*cell-px,i=(x-x0)*2+s;
   xc[i]=dx*co;xs[i]=-dx*si;
  }
  for(let y=y0;y<y1;y++) {
   const dy0=origin[1]+(y+.25)*cell-py,dy1=origin[1]+(y+.75)*cell-py;
   const ys0=dy0*si,ys1=dy1*si,yc0=dy0*co,yc1=dy1*co;
   for(let x=x0;x<x1;x++) {
    // Only omit cells wholly inside the existing canvas clip hole.
    if(hidden && origin[0]+x*cell>=hidden[0] && origin[0]+(x+1)*cell<=hidden[2]
       && origin[1]+y*cell>=hidden[1] && origin[1]+(y+1)*cell<=hidden[3])continue;
    let area=0;const i=(x-x0)*2;
    for(let s=0;s<2;s++) {
     const a0=xc[i+s]+ys0,a1=xc[i+s]+ys1;
     if(a0>=0&&a0<=g.length&&Math.abs(xs[i+s]+yc0)<=g.half)area+=.25;
     if(a1>=0&&a1<=g.length&&Math.abs(xs[i+s]+yc1)<=g.half)area+=.25;
    }
    coverage[y*width+x]=Math.max(coverage[y*width+x],area);
   }
  }
 }
 return coverage;
}
function powerItemGrid(entities) {
 const amounts=new Float32Array(48*56),raw={1:1,4:20,10:1,11:20};
 for(const e of entities||[]){
  const amount=raw[e.type];if(!amount)continue;
  const [x,y]=e.position;if(x<0||x>=384||y<0||y>=448)continue;
  amounts[Math.floor(y/8)*48+Math.floor(x/8)]+=amount;
 }
 for(let i=0;i<amounts.length;i++)amounts[i]/=100;
 return amounts;
}
// Display contrast only; powerItemGrid and policy inputs retain linear amounts.
function powerDisplayAlpha(amount) {
 return amount>0?Math.min(1,.4+.6*Math.sqrt(amount/.2)):0;
}
function paintCoverage(c,coverage,origin,cell,width,color,opacity=1) {
 c.fillStyle=color;
 for(let i=0;i<coverage.length;i++)if(coverage[i]>0) {
  c.globalAlpha=coverage[i]*opacity;c.fillRect(origin[0]+i%width*cell,origin[1]+Math.floor(i/width)*cell,cell,cell);
 }
 c.globalAlpha=1;
}
// Hold/final packets use the generic nearest-bullet display payload even for
// dual-grid training. The explicit learning/model contract is authoritative.
function observationFrame(f) {
 const contract=f.learning?.contract??f.model?.contract;
 const dual=['th10-dual-grid-v2','th10-dual-grid-v3','th10-dual-grid-v4','th10-dual-grid-v5','th10-dual-grid-v6','th10-dual-grid-v6-action-grid-v1'].includes(contract)||(!contract&&(f.model?.bullet_scope?.shape??f.ai_observation?.bullet_scope?.shape)==='dual_grid');
 if(!dual)return f;
 return {...f,ai_observation:{...f.ai_observation,bullet_scope:{shape:'dual_grid'},
  bullets:f.entities?.bullets??null,items:f.entities?.items??null,
  local_viewport:{center:f.player?.position??null}}};
}
const focusedField = field;
field = function(f) {
 if(compact)return; // obs2 has no visible field; avoid all grid reconstruction.
 const contract=f.learning?.contract??f.model?.contract;
 const motionOffsets=['th10-dual-grid-v4','th10-dual-grid-v5','th10-dual-grid-v6','th10-dual-grid-v6-action-grid-v1'].includes(contract),powerLayer=['th10-dual-grid-v5','th10-dual-grid-v6','th10-dual-grid-v6-action-grid-v1'].includes(contract);
 const legend=document.querySelector('.observation-key');
 if(legend) {
  const shape=f.ai_observation?.bullet_scope?.shape,dual=shape==='dual_grid',nearest=shape==='nearest';
  legend.children[0].textContent=dual?(powerLayer?'全体：8px／青：自機弾／橙：P量':'全体：8px単位／青：自機弾'):nearest?'個別弾：最大40発':'観測構成を取得中';
  legend.children[1].textContent=dual?'枠内192px：弾の判定を2px単位':nearest?'至近10発：個別経路':'—';
  legend.title=dual?(motionOffsets?'表示用の再描画。弾は現在位置と観測速度で2F・4F分ずらした位置を濃淡表示（4Fほど薄い）。HPなどの数値層は省略。P量は小Pも見えるよう表示の濃さを補正（モデル入力は線形）。':'表示用の再描画。速度・HPなどの数値層は省略。P量は小Pも見えるよう表示の濃さを補正（モデル入力は線形）。'):'';
 }
 if (f.ai_observation?.bullet_scope?.shape !== 'dual_grid') return focusedField(f);
 const c=$('field').getContext('2d');c.clearRect(0,0,384,448);
 c.imageSmoothingEnabled=false;
 const center=f.ai_observation.local_viewport?.center;
 const lasers=prepareLasers(f.entities?.lasers);
 const hidden=center?[center[0]-96,center[1]-96,center[0]+96,center[1]+96]:null;
 for (const [kind,list] of Object.entries(f.entities||{})) {
  c.fillStyle=c.strokeStyle={bullets:'#b4e4e9',enemies:'#edc17e',items:'#88d7ac',lasers:'#e9a1ca',player_shots:'#82aaff'}[kind];
  if(kind==='bullets'||kind==='lasers') {
   c.save();if(center){c.beginPath();c.rect(0,0,384,448);c.rect(center[0]-96,center[1]-96,192,192);c.clip('evenodd');}
  }
  if(kind==='bullets'&&motionOffsets)for(const [frames,alpha] of [[4,.2],[2,.4]])for(const e of list||[]){
   if(!(e.flags_raw&2)||!Array.isArray(e.velocity_raw)||e.velocity_raw.length!==2||!e.velocity_raw.every(Number.isFinite))continue;
   const x=e.position[0]+frames*e.velocity_raw[0],y=e.position[1]+frames*e.velocity_raw[1];
   if(x<0||x>=384||y<0||y>=448)continue;
   c.globalAlpha=alpha;c.fillRect(Math.floor(x/8)*8,Math.floor(y/8)*8,8,8);c.globalAlpha=1;
  }
  if(kind==='lasers')paintCoverage(c,gridLaserCoverage(list,[0,0],8,48,56,lasers,hidden),[0,0],8,48,'#e9a1ca');
  for(const e of list||[]) {
   if(kind==='lasers')continue;
   if(kind==='items'&&powerLayer&&[1,4,10,11].includes(e.type))continue;
   if(kind==='player_shots'){const [w,h]=e.hitbox_raw;c.globalAlpha=.5;c.fillRect(e.position[0]-w/2,e.position[1]-h/2,w,h);c.globalAlpha=1;continue;}
   if(kind==='bullets'&&Number.isInteger(e.flags_raw)&&!(e.flags_raw&2))continue;
   const [x,y]=e.position;c.globalAlpha=.65;
   c.fillRect(Math.floor(x/8)*8,Math.floor(y/8)*8,8,8);c.globalAlpha=1;
  }
  if(kind==='bullets'||kind==='lasers')c.restore();
 }
 if(powerLayer){
  const amounts=powerItemGrid(f.entities?.items);c.fillStyle='#ffa64d';
  for(let i=0;i<amounts.length;i++)if(amounts[i]>0){c.globalAlpha=powerDisplayAlpha(amounts[i]);c.fillRect(i%48*8,Math.floor(i/48)*8,8,8);}
  c.globalAlpha=1;
 }
 if(center) {
  const [x,y]=center,origin=[x-96,y-96];
  if(motionOffsets)for(const [frames,alpha] of [[4,.2],[2,.4]])paintCoverage(c,localBulletCoverage(f.entities?.bullets,center,frames),origin,2,96,'#b4e4e9',alpha);
  paintCoverage(c,localBulletCoverage(f.entities?.bullets,center),origin,2,96,'#b4e4e9');
  paintCoverage(c,gridLaserCoverage(f.entities?.lasers,origin,2,96,96,lasers),origin,2,96,'#e9a1ca');
  c.strokeStyle='#eeb956';c.lineWidth=1;c.strokeRect(x-96,y-96,192,192);
  c.fillStyle='#d4ffed';c.fillRect(x-1,y-1,2,2);
 }
};
function node(parent,tag,text,cls=''){const e=document.createElement(tag);e.textContent=text;e.className=cls;parent.append(e);return e;}
for(const d of order){const e=node($('directions'),'div','','direction');e.dataset.d=d;const i=node(e,'i','');i.dataset.lucide=icons[d];}
for(const [key,label] of [['focus','低速'],['shoot','ショット'],['bomb','ボム']]){const e=node($('buttons'),'div','',`button ${key}`);e.dataset.key=key;node(e,'span',label);node(e,'b','—');}
lucide.createIcons();
for(const label of Object.values(labels)){const r=node($('weights'),'div','','row');node(r,'span',label);node(r,'strong','—');}
function direction(mask){const u=mask&16,d=mask&32,l=mask&64,r=mask&128;return u?(r?2:l?8:1):d?(r?4:l?6:5):r?3:l?7:0;}
function field(f){const c=$('field').getContext('2d');c.clearRect(0,0,384,448);for(const cell of f.ai_observation?.bullet_grid||[]){const [x,y]=cell.position;c.fillStyle=`rgba(238,185,86,${.2+.8*cell.density})`;c.fillRect(x,y,4,4);c.strokeStyle='#eeb956';c.lineWidth=.7;c.beginPath();c.moveTo(x+2,y+2);c.lineTo(x+2+cell.velocity[0]*8,y+2+cell.velocity[1]*8);c.stroke();}for(const [kind,list] of Object.entries({...f.entities,bullets:f.ai_observation?.bullets??[],items:f.ai_observation?.items??[]})){c.fillStyle=c.strokeStyle={bullets:'#b4e4e9',enemies:'#edc17e',items:'#88d7ac',lasers:'#e9a1ca',player_shots:'#82aaff'}[kind];for(const [index,e] of (list||[]).entries()){if(kind==='lasers'){const g=e.collision;if(!g?.active)continue;c.save();c.translate(g.origin[0]+192,g.origin[1]);c.rotate(g.angle);c.globalAlpha=.35;c.fillRect(0,-g.width/2,g.length,g.width);c.globalAlpha=1;c.strokeRect(0,-g.width/2,g.length,g.width);c.restore();continue;}const [x,y]=e.position;if(kind==='enemies')c.strokeRect(x-5,y-5,10,10);else{c.fillStyle=kind==='bullets'?(index<10?'#eeb956':'#b4e4e9'):'#88d7ac';c.beginPath();c.arc(x,y,kind==='bullets'?2:2.8,0,Math.PI*2);c.fill();}}}if(f.player){const [x,y]=f.player.position;c.strokeStyle='#88d7ac';c.strokeRect(x-5,y-5,10,10);c.fillStyle='#d4ffed';c.fillRect(x-1,y-1,2,2);}}
let rewardEpisode=null, liveEpisodeReturn=null;
function renderEpisodeReturn(f) {
 if(f?.episode_id && f.learning) {
  if(f.episode_id!==rewardEpisode) {
   rewardEpisode=f.episode_id;
   liveEpisodeReturn=0;
  }
  if(Number.isFinite(f.reward?.episode_return))liveEpisodeReturn=f.reward.episode_return;
 }
 $('episodeReturn').textContent=fmt(liveEpisodeReturn);
}
function render(f,fresh){renderEpisodeReturn(f);f=observationFrame(f);$('stage').textContent=`STAGE ${f.game?.stage??'—'}`;$('objects').textContent=`${f.ai_observation?.bullets?.length??'—'} BULLETS`;field(f);const p=f.policy,mask=fresh?f.applied_input:null;const best=p?.directions?.indexOf(Math.max(...p.directions));for(const e of $('directions').children){e.classList.toggle('predicted',fresh&&+e.dataset.d===best);e.classList.toggle('pressed',Number.isInteger(mask)&&+e.dataset.d===direction(mask));e.title=p?`${fmt(p.directions[+e.dataset.d]*100,1)}%`:'';}for(const e of $('buttons').children){e.querySelector('b').textContent=p?`${(!compact&&e.dataset.key==='bomb'?fmt(p.bomb*100,2):fmt(p[e.dataset.key]*100,0))}%`:'—';e.classList.toggle('pressed',Number.isInteger(mask)&&!!(mask&{shoot:1,focus:4,bomb:2}[e.dataset.key]));}$('value').textContent=fresh?fmt(p?.value):'—';$('weights').replaceChildren();for(const [key,label] of Object.entries(labels)){const r=node($('weights'),'div','','row');node(r,'span',label);const w=f.model?.reward_weights?.[key];node(r,'strong',Number.isFinite(w)?`${w>0?'+':''}${fmt(w)} / ${key==='hit'?'被弾':key==='bomb'?'発動':'イベント'}`:'未接続',Number.isFinite(w)?'':'muted');}}
function plot(id,points,key,color,start,end){const el=$(id),c=el.getContext('2d'),w=el.width,h=el.height;c.clearRect(0,0,w,h);const ys=points.map(p=>p[key]).filter(Number.isFinite);const lo=Math.min(0,...ys),hi=Math.max(0,...ys),range=hi-lo||1;c.font='11px sans-serif';for(let i=0;i<3;i++){const y=14+i*(h-30)/2;c.strokeStyle='#ffffff20';c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke();c.fillStyle='#aeb5c2';c.fillText(fmt(hi-i*range/2,1),2,y-3);}c.strokeStyle=color;c.lineWidth=1.5;c.beginPath();let active=false;for(const p of points){if(!Number.isFinite(p[key])){active=false;continue;}const x=38+(p.time-start)/Math.max(1,end-start)*(w-42),y=14+(hi-p[key])/range*(h-30);if(active)c.lineTo(x,y);else c.moveTo(x,y);active=true;}c.stroke();}

// Same episode index for all series. Current-version missing progress is Start (0).
function drawGrowth(el, episodes, progressAxis=[]) {
 const c=el.getContext('2d'),w=el.width,h=el.height,L=44,R=w-94,T=22,B=h-18;
 c.clearRect(0,0,w,h);c.font='11px sans-serif';c.textAlign='left';
 const values=episodes.map(e=>e.return).filter(Number.isFinite),lo=Math.min(0,...values),hi=Math.max(0,...values),range=hi-lo||1;
 const ranks=episodes.map(e=>e.max_progress?.rank).filter(v=>Number.isFinite(v)&&v>=1&&v<=24);
 const top=Math.min(24,Math.max(4,...ranks));
 const x=i=>L+i/Math.max(1,episodes.length-1)*(R-L),rewardY=v=>T+(hi-v)/range*(B-T);
 c.fillStyle='#aeb5c2';c.fillText('報酬',2,11);c.fillStyle='#79d8c4';c.fillText('到達点（右軸）',R+8,11);
 for(let i=0;i<3;i++){const v=hi-i*range/2,y=T+i*(B-T)/2;c.strokeStyle='#ffffff20';c.beginPath();c.moveTo(L,y);c.lineTo(R,y);c.stroke();c.fillStyle='#aeb5c2';c.fillText(fmt(v,0),2,y+4);}
 const names=['中ボス着','中ボス突破','ボス着','ボス突破'];
 // Keep every milestone visible, including midboss clears.
 const axis=progressAxis.length?progressAxis:Array.from({length:24},(_,i)=>({rank:i+1,label:`${Math.floor(i/4)+1}面 ${names[i%4]}`}));
 const available=[{rank:0,label:'スタート'},...axis.filter(p=>p.rank<=top)];
 // Each milestone occupies one equal display interval, including spell clears.
 // Stored ranks still identify progression; their fractional gaps are not distances.
 const py=v=>{
  let index=available.findIndex(p=>p.rank>=v);
  if(index<0)index=available.length-1;
  const lower=available[Math.max(0,index-1)],upper=available[index];
  const position=index===0?0:index-1+(v-lower.rank)/(upper.rank-lower.rank);
  return B-position/Math.max(1,available.length-1)*(B-T);
 };
 c.font=`${Math.min(11,(B-T)/Math.max(1,available.length-1)*.85)}px sans-serif`;
 c.fillStyle='#79d8c4';for(const p of available){const y=py(p.rank);c.fillText(p.label.replace(/ボス終了/g,'ボス突破'),R+8,y+4);}
 c.font='11px sans-serif';
 function line(data,color,y,width,dots=false,connected=true){c.strokeStyle=c.fillStyle=color;c.lineWidth=width;if(connected){c.beginPath();let active=false;data.forEach((v,i)=>{if(!Number.isFinite(v)){active=false;return;}if(active)c.lineTo(x(i),y(v));else c.moveTo(x(i),y(v));active=true;});c.stroke();}if(dots)data.forEach((v,i)=>{if(Number.isFinite(v)){c.beginPath();c.arc(x(i),y(v),2.5,0,Math.PI*2);c.fill();}});}
 // Draw progress behind reward points. A baseline mark keeps Start visible.
 const barWidth=Math.min(12,(R-L)/Math.max(1,episodes.length-1)*.7);
 c.fillStyle='rgba(121,216,196,0.35)';
 episodes.forEach((e,i)=>{const r=e.max_progress?.rank,rank=Number.isFinite(r)&&r>=1&&r<=24?r:0;
  const left=Math.max(L,x(i)-barWidth/2),right=Math.min(R,x(i)+barWidth/2),y=Math.min(B-2,py(rank));
  c.fillRect(left,y,right-left,B-y);
 });
 line(episodes.map(e=>e.return),'#b0bbc9',rewardY,1.3,true,false);
 let rewardStart=0;
 const rewardAverage=episodes.map((e,i)=>{
  if(i && e.reward_version!==episodes[i-1].reward_version){
   rewardStart=i;c.strokeStyle='#e6bb5c80';c.beginPath();c.moveTo(x(i),T);c.lineTo(x(i),B);c.stroke();
   c.fillStyle='#e6bb5c';c.fillText('報酬変更',Math.min(x(i)+3,R-52),T-5);
  }
  const a=episodes.slice(Math.max(rewardStart,i-19),i+1).map(e=>e.return).filter(Number.isFinite);
  return i===rewardStart && i>0?null:(a.length?a.reduce((a,b)=>a+b,0)/a.length:null);
 });
 line(rewardAverage,'#e6bb5c',rewardY,2);

}

function renderStats(s){const rewardMax=5;$('rewardScale').textContent=`直近30秒 / 共通上限 |${fmt(rewardMax)}|`;$('totals').replaceChildren();for(const [key,label] of Object.entries(labels)){const row=node($('totals'),'div','','row');node(row,'span',label);const bar=node(row,'meter','','reward-bar '+key);bar.min=0;bar.max=rewardMax;bar.value=Number.isFinite(s.totals[key])?Math.abs(s.totals[key]):0;bar.setAttribute('aria-label',label+' 絶対値 / 共通上限 '+fmt(rewardMax));node(row,'strong',Object.hasOwn(s.totals,key)?fmt(s.totals[key]):'未接続',Object.hasOwn(s.totals,key)?'reward-number':'muted');}$('td').textContent=fmt(s.td_mean);$('samples').textContent=`${s.samples} 観測`;$('window').textContent=`${new Date(s.start*1000).toLocaleTimeString('ja-JP')} — ${new Date(s.end*1000).toLocaleTimeString('ja-JP')}`;plot('values',s.points,'value','#e6bb5c',s.start,s.end);plot('errors',s.points,'td','#8bcede',s.start,s.end);const growth=s.growth;drawGrowth($('growth'),growth,s.progress_axis);$('first').textContent=growth.length?'1回目':'記録なし';$('last').textContent=growth.length?`${growth.length}回目`:'';}
const source=new EventSource('/api/live-stream');source.onmessage=e=>{try{pending=JSON.parse(e.data);}catch{}};
let previousFresh=null;
function synchronizedSummary(){
 const points=stats?.points||[];$('value').textContent=fmt(points.length?points[points.length-1].value:null);
 const w=stats?.weights||{},signed=v=>Number.isFinite(v)?`${v>0?'+':''}${fmt(v,1)}`:'—';
for(const [i,key] of Object.keys(labels).entries()){const e=$('weights').children[i]?.querySelector('strong');if(!e)continue;const texts={damage:`${fmt(w.damage)} + ${fmt(w.damage*w.damage_power,3)} × Power / 1000HPごと・ボム中0`,progress:`${fmt(w.progress)} + ${fmt(w.progress_life)} × 残機 + ${fmt(w.progress_power)} × Power・ボム中0`,hit:`${signed(w.hit)} / 1回`,jitter:stats?.reward_version==='th10-rewards-v23'?`${signed(w.jitter)} × min(振動量 / 4, 1) / 秒・移動変化時`:`${signed(w.jitter)} / 往復判定1回（最短12F間隔）`,power_gain:`${fmt(w.power_gain)} × 獲得Power・ボム中0`};e.textContent=texts[key];e.classList.toggle('muted',!(stats?.enabled||[]).includes(key));}
}
function tick(){let changed=false;if(pending){frame=pending;pending=null;changed=true;}const fresh=!!frame&&Date.now()/1000-frame.timestamp<2.5;$('status').textContent=fresh?(frame.learning?.phase==='playing'?'LIVE / 実機制御':'LIVE / 観測中'):'停止 / 最終観測';const banner=learningBannerState(frame,stats,fresh);if($('learningBanner').textContent!==banner.text)$('learningBanner').textContent=banner.text;$('learningBanner').classList.toggle('playing',banner.playing);if(frame&&(changed||fresh!==previousFresh)){render(frame,fresh);synchronizedSummary();}previousFresh=fresh;requestAnimationFrame(tick);}requestAnimationFrame(tick);
async function poll(){try{const r=await fetch('/api/obs');if(!r.ok)throw Error();stats=await r.json();renderStats(stats);synchronizedSummary();}catch{$('samples').textContent='集計接続待ち';}finally{setTimeout(poll,1000);}}poll();

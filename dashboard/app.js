'use strict';
const $ = id => document.getElementById(id);
const state = {records: [], mode:'recording', index:0, total:0, playing:false, view:'monitor', frame:null, series:[], request:0, busy:false};
const colors = {bullets:'#b4e4e9',enemies:'#edc17e',items:'#88d7ac',lasers:'#e9a1ca'};
const names = {bullets:'弾',enemies:'敵',items:'アイテム',lasers:'レーザー判定'};
const visible = {bullets:true,enemies:true,items:true,lasers:true};
const directions = ['circle','arrow-up','arrow-up-right','arrow-right','arrow-down-right','arrow-down','arrow-down-left','arrow-left','arrow-up-left'];
const order = [8,1,2,7,0,3,6,5,4];
const rewardNames = {damage:'敵ダメージ',kill:'敵撃破',stage_clear:'面クリア',remaining_life:'残機ボーナス',hit:'被弾',bomb:'ボム'};
function icons(){lucide.createIcons();}
function text(id,value){$(id).textContent=value;}
function value(n,digits=0){return Number.isFinite(n)?n.toFixed(digits):'—';}
function duration(n){return Number.isFinite(n)?`${Math.floor(n/60)}:${String(Math.floor(n%60)).padStart(2,'0')}`:'—';}
function report(error){$('error').hidden=false;text('error',error.message||String(error));}
async function get(url){const response=await fetch(url);if(!response.ok)throw new Error(`データを取得できません (${response.status})`);return response.json();}
function cell(parent,tag,content,className=''){const node=document.createElement(tag);node.textContent=content;node.className=className;parent.append(node);return node;}
function iconButton(icon,label,fn){const button=document.createElement('button');button.className='icon-button';button.title=label;button.setAttribute('aria-label',label);const i=document.createElement('i');i.dataset.lucide=icon;button.append(i);button.onclick=fn;return button;}
function buildGrid(id,probabilities=false){for(const direction of order){const box=document.createElement('div');box.className='direction-cell';box.dataset.direction=direction;const i=document.createElement('i');i.dataset.lucide=directions[direction];box.append(i);if(probabilities)cell(box,'span','—');box.title=['静止','上','右上','右','右下','下','左下','左','左上'][direction];$(id).append(box);}}
function canvas(id){const element=$(id);const rect=element.getBoundingClientRect();const ratio=devicePixelRatio||1;if(rect.width===0||rect.height===0)return null;const width=Math.round(rect.width*ratio),height=Math.round(rect.height*ratio);if(element.width!==width||element.height!==height){element.width=width;element.height=height;}const context=element.getContext('2d');context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,rect.width,rect.height);return {context,w:rect.width,h:rect.height};}
function drawLaserCollision(c, collision){
if(!collision||collision.active!==true)return false;
const {origin,angle,length,width}=collision;
if(!Array.isArray(origin)||origin.length!==2||![...origin,angle,length,width].every(Number.isFinite)||length<=0||width<=0)return false;
c.save();
// Collision origins use native coordinates, unlike the already shifted entity positions.
c.translate(origin[0]+192,origin[1]);c.rotate(angle);
c.globalAlpha=.35;c.fillRect(0,-width/2,length,width);
c.globalAlpha=1;c.lineWidth=.6;c.strokeRect(0,-width/2,length,width);
c.restore();return true;
}
function drawField(){const target=canvas('field');if(!target)return;const {context:c,w,h}=target;c.fillStyle='#101a18';c.fillRect(0,0,w,h);c.save();c.scale(w/384,h/448);c.strokeStyle='#24322c';c.lineWidth=.6;for(let x=0;x<=384;x+=32){c.beginPath();c.moveTo(x,0);c.lineTo(x,448);c.stroke();}for(let y=0;y<=448;y+=32){c.beginPath();c.moveTo(0,y);c.lineTo(384,y);c.stroke();}c.setLineDash([3,5]);c.strokeStyle='#43574c';c.beginPath();c.moveTo(192,0);c.lineTo(192,448);c.stroke();c.setLineDash([]);
const frame=state.frame;if(!frame){c.restore();return;}for(const [kind,entities] of Object.entries(frame.entities)){if(!visible[kind]||!entities)continue;c.strokeStyle=colors[kind];c.fillStyle=colors[kind];for(const entity of entities){if(kind==='lasers'){drawLaserCollision(c,entity.collision);continue;}const [x,y]=entity.position;if(x< -50||x>434||y< -100||y>500)continue;c.beginPath();if(kind==='bullets'){c.arc(x,y,2.3,0,Math.PI*2);c.fill();}else if(kind==='enemies'){c.strokeRect(x-6,y-6,12,12);c.moveTo(x-10,y);c.lineTo(x+10,y);c.moveTo(x,y-10);c.lineTo(x,y+10);c.stroke();}else if(kind==='items'){c.moveTo(x,y-3);c.lineTo(x+3,y);c.lineTo(x,y+3);c.lineTo(x-3,y);c.closePath();c.fill();}}}
if(frame.player){const [x,y]=frame.player.position;c.strokeStyle='#8dccaa';c.lineWidth=.8;c.beginPath();c.arc(x,y,28,0,Math.PI*2);c.stroke();c.fillStyle='#a6f2c2';c.beginPath();c.moveTo(x,y-7);c.lineTo(x+5,y+5);c.lineTo(x,y+2);c.lineTo(x-5,y+5);c.closePath();c.fill();}c.restore();}
function chart(id,points,key,color){const target=canvas(id);if(!target)return;const {context:c,w,h}=target;const max=Math.max(1,...points.map(p=>p[key]||0));c.font='10px sans-serif';for(let k=0;k<3;k++){const y=12+(h-30)*k/2;c.strokeStyle='#2b3730';c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke();c.fillStyle='#8da092';c.fillText(String(Math.round(max*(1-k/2))),3,y-3);}c.strokeStyle=color;c.lineWidth=1.5;c.beginPath();let started=false;points.forEach((p,i)=>{if(p[key]===null||p[key]===undefined){started=false;return;}const x=20+i/Math.max(1,points.length-1)*(w-24),y=h-18-p[key]/max*(h-30);if(started)c.lineTo(x,y);else c.moveTo(x,y);started=true;});c.stroke();if(id==='bulletChart'&&state.total){const x=20+state.index/Math.max(1,state.total-1)*(w-24);c.strokeStyle='#b9d6c2';c.setLineDash([2,3]);c.beginPath();c.moveTo(x,8);c.lineTo(x,h-12);c.stroke();c.setLineDash([]);}}
function drawHistory(){chart('historyChart',state.records.slice().reverse(),'stage_frame','#92dfb2');}
function drawInput(mask){const valid=Number.isInteger(mask);let d=0;if(valid){const up=!!(mask&16),down=!!(mask&32),left=!!(mask&64),right=!!(mask&128);d=up?(right?2:left?8:1):down?(right?4:left?6:5):right?3:left?7:0;}for(const box of $('inputGrid').children)box.classList.toggle('active',valid&&Number(box.dataset.direction)===d);for(const [id,bit] of [['shootInput',1],['focusInput',4],['bombInput',2]]){const active=valid&&!!(mask&bit);$(id).classList.toggle('on',active);$(id).querySelector('b').textContent=valid?(active?'ON':'OFF'):'未記録';}text('inputStatus',valid?'実測':'未記録');}
function render(frame){state.frame=frame;const game=frame.game;const counts=Object.fromEntries(Object.entries(frame.entities).map(([k,v])=>[k,v===null?null:v.length]));text('stage',game.stage===null?'—':`STAGE ${game.stage}`);text('elapsed',duration(frame.metrics.stage_seconds));text('lives',value(game.lives_reserve));text('power',value(game.power,2));text('character',`${['Easy','Normal','Hard','Lunatic','Extra'][game.difficulty]||'—'} / ${['霊夢','魔理沙'][game.character]||'—'} ${['A','B','C'][game.shot]||''}`);text('frame',value(frame.game_frame));text('position',frame.player?frame.player.position.map(n=>value(n,1)).join(' , '):'—');text('distance',Number.isFinite(frame.metrics.nearest_bullet_center_pixels)?`${value(frame.metrics.nearest_bullet_center_pixels,1)} px`:'—');text('latency',Number.isFinite(frame.metrics.sample_ms)?`${value(frame.metrics.sample_ms,1)} ms`:'—');text('sync',frame.capabilities.frame_locked?'フレーム固定':'未検証');text('objectCount',`${counts.bullets===null?'—':counts.bullets} BULLETS`);for(const kind of Object.keys(names))text(`count-${kind}`,counts[kind]===null?'—':counts[kind]);drawInput(frame.applied_input);text('sequence',`sequence ${frame.sequence??'—'}`);text('rawJson',JSON.stringify(frame,null,2));$('fieldEmpty').hidden=!!frame.player;text('cursor',`${state.index+1} / ${state.total}`);$('seek').value=state.index;text('updated',new Date().toLocaleTimeString('ja-JP'));drawField();chart('bulletChart',state.series,'bullets','#edc17e');}
function renderLearning(frame){
const policy=frame?.policy, reward=frame?.reward, learning=frame?.learning;
const bosses=(frame?.entities.enemies||[]).filter(e=>e.is_boss);
text('observedBoss',bosses.length?bosses.map(e=>`${value(e.hp)} / ${value(e.hp_max)}`).join(', '):'—');
text('observedSpell',frame?.spell?((frame.spell.flags_raw&1)?`スペル ID ${frame.spell.id_raw}`:bosses.length?'通常攻撃':'非スペル'):'—');
text('observedBomb',frame?.bomb?`状態 ${frame.bomb.state} / ${frame.bomb.timer} frames`:'—');
const accelerations=(frame?.entities.bullets||[]).filter(b=>Array.isArray(b.acceleration));
text('observedAcceleration',frame?.entities.bullets?`${accelerations.length} / ${frame.entities.bullets.length} 弾`:'—');
document.querySelector('.metrics-strip>div:last-child strong').textContent=learning?String(learning.steps):frame?.model?`${frame.model.steps} (保存済み)`:'未接続';
const section=document.querySelector('.policy-section');const headings=section.querySelectorAll('.section-heading .tag');
headings[1].textContent=policy?(policy.source==='real_observation_shadow'?'実観測推論 / 入力なし':'実機制御中'):'未接続';headings[2].textContent=reward?reward.enabled.map(k=>rewardNames[k]||k).join(' / '):'イベント未接続';
const outputs=section.querySelectorAll('.data-list dd');outputs[0].textContent=policy?value(policy.value,3):'—';outputs[1].textContent=policy?`射撃 ${value(policy.shoot*100,1)}% / 低速 ${value(policy.focus*100,1)}% / ボム ${value(policy.bomb*100,1)}%`:'—';outputs[2].textContent=learning?String(learning.updates):policy?`${policy.trained_updates} (保存済み)`:'—';
for(const box of $('realProbGrid').children)box.querySelector('span').textContent=policy?`${value(policy.directions[Number(box.dataset.direction)]*100,1)}%`:'—';
for(const [i,key] of Object.keys(rewardNames).entries())$('rewardRows').children[i].querySelector('strong').textContent=reward?.enabled.includes(key)?value(reward.components[key],2):'—';
section.querySelector('.reward-total strong').textContent=reward?value(reward.episode_return,2):'—';section.querySelector('.reward-total span').textContent='試行の累積報酬';
const strips=document.querySelectorAll('.interface-strip span');strips[2].textContent=`policy: ${policy?.source||'null'}`;strips[3].textContent=`reward: ${reward?reward.enabled.join(','):'null'}`;
const coverage=document.querySelectorAll('.coverage>div>span:last-child');
coverage[1].textContent=frame?.capabilities.enemy_hp?'実測 / AI観測接続済み':'現在の観測値なし';
coverage[2].textContent=frame?.capabilities.acceleration?'差分加速度接続 / 継続IDなし':'加速度は連続観測待ち / 継続IDなし';
coverage[3].textContent=frame?.capabilities.laser_geometry_validated?'直線判定は実測照合済み / 発生・消失は未検証':frame?.capabilities.laser_geometry_available?'解析判定を接続 / 今回の実測範囲外':'現在レーザーなし';
if(!learning)text('recordHint',policy?.source==='real_observation_shadow'?'実モデル推論 / 操作・学習なし':'観測のみ / AI制御なし');
renderActualModel(frame);
if(learning)text('recordHint',({'playing':'実機プレイ中','optimizing_at_game_over':'ゲームオーバー画面で学習更新中','game_over_ready':'学習更新完了 / 継続待機'})[learning.phase]||'実機学習 / 限定プロファイル');
}
const renderObservation=render;render=function(frame){renderObservation(frame);renderLearning(frame);};
const measurements=cell(document.querySelector('.policy-section'),'section','','observed-measurements');
cell(measurements,'h2','追加取得値 / 検証中');const measuredList=cell(measurements,'dl','','data-list');
for(const [id,label] of [['observedBoss','ボスHP / 最大HP'],['observedSpell','ボス状態'],['observedBomb','ボム取得値'],['observedAcceleration','加速度取得数']]){cell(measuredList,'dt',label);cell(measuredList,'dd','—').id=id;}
const realProbGrid=document.createElement('div');realProbGrid.id='realProbGrid';realProbGrid.className='direction-grid probability';realProbGrid.setAttribute('aria-label','実機ポリシーの移動確率');document.querySelector('.policy-section .data-list').after(realProbGrid);buildGrid('realProbGrid',true);
const realHistory=document.createElement('section');realHistory.className='raw-section';
const realTitle=cell(realHistory,'h2','実機学習ラン');const realRows=cell(realHistory,'div','','training-list');
$('history').prepend(realHistory);
document.querySelector('#model .page-heading .tag').textContent='モデル接続待ち';
document.querySelector('#model h1').textContent='実モデル・学習設定';
async function refreshLearning(){try{const data=await get('/api/live-learning');realRows.replaceChildren();for(const run of data.runs){const row=cell(realRows,'div','');cell(row,'span',run.run);cell(row,'span',`${run.status} / ${run.gameplay_training_steps} steps / ${run.updates} updates / 限定プロファイル`);for(const ep of run.episodes){const detail=cell(realRows,'div','');cell(detail,'span',`試行 ${ep.episode} / ${ep.steps} steps`);cell(detail,'span',`報酬 ${value(ep.return,2)} / 被弾 ${ep.hits} / ${ep.terminated?'ゲームオーバー':'時間上限'}`);}}}catch(e){report(e);}}
refreshLearning();setInterval(refreshLearning,5000);
function stop(){state.playing=false;$('play').replaceChildren();const i=document.createElement('i');i.dataset.lucide='play';$('play').append(i);$('play').title='再生';$('play').setAttribute('aria-label','再生');icons();}
async function seek(index){if(state.mode!=='recording'||!state.records.length)return;const request=++state.request;state.index=Math.max(0,Math.min(state.total-1,index));const frame=await get(`/api/frame?id=${encodeURIComponent($('recording').value)}&index=${state.index}`);if(request!==state.request)return;render(frame);}
async function selectRecord(id){stop();state.request++;$('recording').value=id;const record=state.records.find(r=>r.id===id);if(!record)return;state.total=record.frames;$('seek').max=record.frames-1;text('runStatus',record.terminated?'終了 / ゲームオーバー':record.status==='passed'?'検証済み':'検証記録');const selected=id;const data=await get(`/api/series?id=${encodeURIComponent(id)}`);if($('recording').value!==selected||state.mode!=='recording')return;state.series=data.points;const peak=state.series.reduce((best,p)=>p.bullets>(best.bullets||0)?p:best,{index:0,bullets:0});await seek(peak.index);}
async function catalog(){const data=await get('/api/catalog');state.records=data.recordings;const selected=$('recording').value;$('recording').replaceChildren();for(const r of state.records){const option=document.createElement('option');option.value=r.id;option.textContent=r.id;$('recording').append(option);}$('historyRows').replaceChildren();for(const r of state.records){const tr=document.createElement('tr');for(const v of [r.id,r.status==='passed'?'検証済み':r.status,r.frames,r.stage_frame??'—',r.counts_max.bullets,r.terminated?'ゲームオーバー':'時間上限 / その他'])cell(tr,'td',v);const td=cell(tr,'td','');td.append(iconButton('arrow-up-right','この記録を表示',async()=>{showView('monitor');setMode('recording');await selectRecord(r.id);}));$('historyRows').append(tr);}$('trainingRows').replaceChildren();for(const r of data.training_runs){const row=cell($('trainingRows'),'div','');cell(row,'span',r.run);cell(row,'span',`${r.status} / ${r.timesteps??'—'} steps / 模擬・報酬0`);}if(!state.records.length){text('runStatus','記録なし');$('fieldEmpty').hidden=false;for(const id of ['play','seek','back','next'])$(id).disabled=true;}else if(state.mode==='recording'){await selectRecord(state.records.some(r=>r.id===selected)?selected:(state.records.find(r=>r.counts_max.bullets>0)||state.records[0]).id);}icons();drawHistory();}
function showView(view){state.view=view;for(const element of document.querySelectorAll('.view'))element.hidden=element.id!==view;for(const button of document.querySelectorAll('[data-view]'))button.classList.toggle('active',button.dataset.view===view);if(view!=='monitor')stop();requestAnimationFrame(()=>{drawField();drawHistory();chart('bulletChart',state.series,'bullets','#edc17e');});}
function setMode(mode){stop();state.stream?.close();state.stream=null;state.pending=null;state.request++;state.mode=mode;const live=mode==='live';$('liveMode').setAttribute('aria-pressed',live);$('recordMode').setAttribute('aria-pressed',!live);for(const id of ['recording','back','next','play','seek','speed'])$(id).disabled=live;text('sourceBadge',live?'ライブ接続待ち':'実機記録');text('recordHint',live?'ライブ / 読み取り専用':'実機記録 / 学習なし');if(live){state.total=0;state.series=[];startLiveStream();}else if($('recording').value)selectRecord($('recording').value).catch(report);}
const streamMetric=cell(document.querySelector('.interface-strip'),'span','');streamMetric.id='streamMetric';
function startLiveStream(){
const generation=state.request;state.arrivals=[];state.presentations=[];
state.stream=new EventSource('/api/live-stream');
state.stream.onmessage=event=>{if(state.mode!=='live'||generation!==state.request)return;try{state.pending=JSON.parse(event.data);state.arrivals.push(performance.now());}catch(error){report(error);}};
state.stream.onerror=()=>{if(state.mode==='live')text('sourceBadge','再接続中 / 最終観測');};
function present(){if(state.mode!=='live'||generation!==state.request)return;const now=performance.now();
if(state.pending){const data=state.pending;state.pending=null;state.index=0;state.total=1;text('runStatus',data.episode_id);render(data);state.presentations.push(now);}
state.arrivals=state.arrivals.filter(t=>now-t<2000);state.presentations=state.presentations.filter(t=>now-t<2000);
const fresh=state.frame&&Date.now()/1000-state.frame.timestamp<2.5;
text('sourceBadge',fresh?'ライブ実測':'停止 / 最終観測');text('cursor',fresh?'LIVE':'STOPPED');
streamMetric.textContent=`受信 ${(state.arrivals.length/2).toFixed(1)} Hz / 描画 ${(state.presentations.length/2).toFixed(1)} fps / 経過 ${state.frame?Math.max(0,Date.now()-state.frame.timestamp*1000).toFixed(0):'—'} ms`;
requestAnimationFrame(present);}requestAnimationFrame(present);
}
async function livePoll(){const generation=state.request;if(state.mode!=='live')return;try{const data=await get('/api/live');if(state.mode!=='live'||generation!==state.request)return;if(data.available===false){state.frame=null;text('sourceBadge','未接続');text('runStatus','ライブ取得は停止中');for(const id of ['stage','elapsed','lives','power','character','frame','position','distance','latency','sync','sequence','cursor','objectCount'])text(id,'—');for(const kind of Object.keys(names))text(`count-${kind}`,'—');text('rawJson','null');drawInput(null);$('fieldEmpty').hidden=false;drawField();chart('bulletChart',[],'bullets','#edc17e');}else{const fresh=Number.isFinite(data.timestamp)&&Date.now()/1000-data.timestamp<2.5;state.index=0;state.total=1;text('sourceBadge',fresh?'ライブ実測':'停止 / 最終観測');text('runStatus',data.episode_id);render(data);text('cursor',fresh?'LIVE':'STOPPED');}}catch(e){report(e);}finally{if(state.mode==='live'&&generation===state.request)setTimeout(livePoll,500);}}
function renderActualModel(frame){
const metadata=frame?.model, policy=frame?.policy;
const panels=document.querySelectorAll('#model .model-grid section');
panels[0].querySelector('h2').textContent='実観測のモデル出力';
panels[0].querySelector('.tag').textContent=policy?(policy.source==='real_observation_shadow'?'推論のみ / 入力なし':'実機制御中'):'推論データなし';
panels[1].querySelector('.tag').textContent=metadata?'読込モデルの実設定':'未取得';
panels[2].querySelector('.tag').textContent=metadata?.reward_scope==='next_extended_training'?'次回拡張学習の設定':metadata?'実行設定':'未取得';
text('diagnosticValue',policy?value(policy.value,3):'—');
for(const box of $('probGrid').children)box.querySelector('span').textContent=policy?`${value(policy.directions[Number(box.dataset.direction)]*100,1)}%`:'—';
for(const row of $('probBars').children){const probability=policy?.[row.dataset.key];row.querySelector('meter').value=probability??0;row.querySelector('meter').hidden=!Number.isFinite(probability);row.lastChild.textContent=Number.isFinite(probability)?`${value(probability*100,1)}%`:'—';}
document.querySelector('#model .page-heading .tag').textContent=metadata?(metadata.migrated?'既存モデル移行 / 追加特徴は未学習':metadata.contract):'モデル未接続';
const signature=JSON.stringify(metadata||null);if(state.modelSignature===signature)return;state.modelSignature=signature;
for(const id of ['ppoSettings','rewardSettings'])$(id).replaceChildren();
for(const [key,v] of Object.entries(metadata?.ppo||{})){const row=cell($('ppoSettings'),'div','');cell(row,'dt',key);cell(row,'dd',String(v));}
if(metadata){const row=cell($('ppoSettings'),'div','');cell(row,'dt','rollout / update');cell(row,'dd','実エピソード長 / ゲームオーバー');}
for(const key of Object.keys(rewardNames)){const v=metadata?.reward_weights?.[key];const row=cell($('rewardSettings'),'div','');cell(row,'dt',rewardNames[key]);cell(row,'dd',Number.isFinite(v)?`${v>0?'+':''}${v}`:'未接続');}
const observationLabels={bullets:'近傍弾',far_grid:'遠方グリッド',lasers:'レーザー',enemies:'敵',items:'アイテム',player:'自機'};
const lists=document.querySelectorAll('#model .architecture ul');lists[0].replaceChildren();
for(const [key,label] of Object.entries(observationLabels)){const item=cell(lists[0],'li',label+' ');cell(item,'b',metadata?.observation_shapes?.[key]?.join(' × ')||'—');}
lists[2].children[1].querySelector('b').textContent=metadata?metadata.action_heads.slice(1).map(n=>`${n}択`).join(' / '):'—';
document.querySelector('#model .configuration-status').textContent=metadata?`${metadata.checkpoint||'学習中モデル'} / ${metadata.steps} steps${metadata.migrated?' / 拡張特徴・ボムは追加学習前':''}`:'実モデル情報なし';
}
async function model(){for(const [key,label] of [['shoot','射撃'],['focus','低速'],['bomb','ボム']]){const row=cell($('probBars'),'div','','prob-bar');row.dataset.key=key;cell(row,'span',label);const meter=document.createElement('meter');meter.min=0;meter.max=1;meter.setAttribute('aria-label',label+'確率');row.append(meter);cell(row,'span','—');}renderActualModel(state.frame);}
for(const kind of Object.keys(names)){const label=document.createElement('label');const input=document.createElement('input');input.type='checkbox';input.checked=true;input.addEventListener('change',()=>{visible[kind]=input.checked;drawField();});label.append(input);const swatch=cell(label,'span','','swatch');swatch.style.setProperty('--swatch',colors[kind]);cell(label,'span',names[kind]);const count=cell(label,'b','—');count.id=`count-${kind}`;$('layers').append(label);}buildGrid('inputGrid');buildGrid('probGrid',true);for(const key of Object.keys(rewardNames)){const row=cell($('rewardRows'),'div','','reward-row');cell(row,'span',rewardNames[key]);cell(row,'span','','reward-track');cell(row,'strong','—');}
for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>showView(button.dataset.view);
$('recording').onchange=()=>selectRecord($('recording').value).catch(report);$('seek').oninput=()=>{stop();seek(Number($('seek').value)).catch(report);};$('back').onclick=()=>{stop();seek(state.index-1).catch(report);};$('next').onclick=()=>{stop();seek(state.index+1).catch(report);};$('recordMode').onclick=()=>setMode('recording');$('liveMode').onclick=()=>setMode('live');$('refresh').onclick=()=>{ $('error').hidden=true;catalog().catch(report);};
$('play').onclick=async()=>{if(state.playing){stop();return;}if(state.index>=state.total-1)await seek(0);state.playing=true;$('play').replaceChildren();const i=document.createElement('i');i.dataset.lucide='pause';$('play').append(i);$('play').title='一時停止';$('play').setAttribute('aria-label','一時停止');icons();};
setInterval(async()=>{if(!state.playing||state.busy||state.mode!=='recording')return;state.busy=true;try{const speed=Number($('speed').value);state.playAccumulator=(state.playAccumulator||0)+speed;if(state.playAccumulator>=1){const step=Math.floor(state.playAccumulator);state.playAccumulator-=step;await seek(state.index+step);}if(state.index>=state.total-1)stop();}catch(e){stop();report(e);}finally{state.busy=false;}},1000/30);
$('download').onclick=()=>{if(!state.frame)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.frame,null,2)],{type:'application/json'}));const anchor=document.createElement('a');anchor.href=url;anchor.download=`observation-${state.frame.game_frame}.json`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
new ResizeObserver(()=>{drawField();chart('bulletChart',state.series,'bullets','#edc17e');drawHistory();}).observe(document.querySelector('main'));
async function init(){try{icons();const health=await get('/api/health');text('connection',health.read_only?'ローカル接続':'接続状態不明');await catalog();await model();if(new URLSearchParams(location.search).get('mode')==='live')setMode('live');}catch(e){text('connection','接続エラー');report(e);}}init();

const fs=require('fs'),vm=require('vm'),assert=require('assert');
const src=fs.readFileSync(require('path').join(__dirname,'../dashboard/obs.js'),'utf8');
const context={};vm.runInNewContext(src.slice(src.indexOf('function observationFrame('),src.indexOf('const focusedField = field;'))+';globalThis.normalize=observationFrame;',context);
for(const phase of ['playing','optimizing_at_game_over','optimizing_recovery','stage_one_ready','game_over_ready']){
 const frame={learning:{contract:'th10-dual-grid-v2',phase},ai_observation:{bullet_scope:{shape:'nearest'},bullets:[]},entities:{bullets:[{position:[1,2]}]},player:{position:[3,4]}};
 const result=context.normalize(frame);
 assert.equal(result.ai_observation.bullet_scope.shape,'dual_grid');
 assert.equal(result.ai_observation.bullets,frame.entities.bullets);
 assert.equal(result.ai_observation.local_viewport.center,frame.player.position);
 assert.equal(frame.ai_observation.bullet_scope.shape,'nearest');
}
const empty=context.normalize({learning:{contract:'th10-dual-grid-v2'}});
assert.equal(empty.ai_observation.bullets,null);assert.equal(empty.ai_observation.local_viewport.center,null);
const old={ai_observation:{bullet_scope:{shape:'nearest'}}};assert.equal(context.normalize(old),old);
assert.equal(context.normalize({model:{contract:'th10-dual-grid-v2'}}).ai_observation.bullet_scope.shape,'dual_grid');
console.log('OBS observation: playing, optimizing, waiting, missing entities, legacy and model contract passed');
assert.equal(context.normalize({model:{contract:'th10-dual-grid-v4'}}).ai_observation.bullet_scope.shape,'dual_grid');
const raster={};vm.runInNewContext(src.slice(src.indexOf('function localBulletCoverage('),src.indexOf('function gridLaserCoverage('))+';globalThis.raster=localBulletCoverage;',raster);
const bullets=[-4,4].map(v=>({position:[192,300],velocity_raw:[v,0],hitbox_raw:[4,4],flags_raw:2}));
const current=raster.raster(bullets,[192,300],0),two=raster.raster(bullets,[192,300],2),four=raster.raster(bullets,[192,300],4);
assert.equal(current[47*96+47],1);assert.equal(two[47*96+47],0);
assert.equal(two[47*96+43],1);assert.equal(two[47*96+51],1);
assert.equal(four[47*96+39],1);assert.equal(four[47*96+55],1);
assert.equal(raster.raster([{...bullets[0],flags_raw:0}],[192,300],2).reduce((a,b)=>a+b,0),0);
console.log('OBS offsets: opposed bullets stay separate at 2F/4F; inactive bullets excluded');
assert.equal(context.normalize({model:{contract:'th10-dual-grid-v5'}}).ai_observation.bullet_scope.shape,'dual_grid');
assert.equal(context.normalize({model:{contract:'th10-dual-grid-v6'}}).ai_observation.bullet_scope.shape,'dual_grid');
assert.equal(context.normalize({model:{contract:'th10-dual-grid-v6-action-grid-v1'}}).ai_observation.bullet_scope.shape,'dual_grid');
const power={};vm.runInNewContext(src.slice(src.indexOf('function powerItemGrid('),src.indexOf('function paintCoverage('))+';globalThis.grid=powerItemGrid;',power);
const grid=power.grid([{type:1,position:[192,100]},{type:10,position:[192,100]},
 {type:4,position:[200,100]},{type:11,position:[200,100]},{type:2,position:[192,100]},
 {type:4,position:[384,100]}]);
assert(Math.abs(grid[12*48+24]-.02)<1e-6);assert(Math.abs(grid[12*48+25]-.4)<1e-6);
console.log('OBS P amounts: small/large aliases summed separately from ordinary items');

const contrast={};vm.runInNewContext(src.slice(src.indexOf('function powerDisplayAlpha('),src.indexOf('function paintCoverage('))+';globalThis.alpha=powerDisplayAlpha;',contrast);
assert.equal(contrast.alpha(0),0);
assert(contrast.alpha(.01)>=.5);
assert(contrast.alpha(.01)<contrast.alpha(.02));
assert(contrast.alpha(.02)<contrast.alpha(.2));
assert.equal(contrast.alpha(.2),1);
assert.equal(contrast.alpha(2),1);
console.log('OBS P contrast: single small P visible, quantities ordered, opacity bounded');

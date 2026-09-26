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

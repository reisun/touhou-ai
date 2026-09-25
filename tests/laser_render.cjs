const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/app.js'), 'utf8');
const helper = source.slice(source.indexOf('function drawLaserCollision('), source.indexOf('function drawField('));
const context = vm.createContext({});
vm.runInContext(helper, context);
const calls = [];
const canvas = Object.fromEntries(['save', 'restore', 'translate', 'rotate', 'fillRect', 'strokeRect']
    .map(name => [name, (...args) => calls.push([name, ...args])]));
const rectangle = {origin: [32.09804916381836, 227.20582580566406],
    angle: 1.435217261314392, length: 144, width: 7, active: true};
assert.equal(context.drawLaserCollision(canvas, rectangle), true);
assert.deepEqual(calls.find(row => row[0] === 'translate'), ['translate', rectangle.origin[0]+192, rectangle.origin[1]]);
assert.deepEqual(calls.find(row => row[0] === 'rotate'), ['rotate', rectangle.angle]);
assert.deepEqual(calls.find(row => row[0] === 'fillRect'), ['fillRect', 0, -3.5, 144, 7]);
assert.deepEqual(calls.find(row => row[0] === 'strokeRect'), ['strokeRect', 0, -3.5, 144, 7]);
for (const invalid of [null, {...rectangle, active: false}, {...rectangle, width: NaN},
    {...rectangle, length: -1}, {...rectangle, origin: [0]}]) {
    calls.length = 0;
    assert.equal(context.drawLaserCollision(canvas, invalid), false);
    assert.equal(calls.length, 0);
}
calls.length = 0;
context.drawLaserCollision(canvas, {...rectangle, width: .25});
assert.deepEqual(calls.find(row => row[0] === 'fillRect'), ['fillRect', 0, -.125, 144, .25]);
console.log('Laser rendering: actual collision coordinates, dimensions, and inactive/missing guards passed.');

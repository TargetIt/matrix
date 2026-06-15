'use strict';

const assert = require('node:assert/strict');
const { evaluate } = require('../assets/gemm-policy.js');

const defaultPolicy = evaluate({
  bm: 128,
  bn: 32,
  bk: 8,
  k: 1024,
  warpM: 64,
  warpN: 16,
  threadM: 8,
  threadN: 4,
  buffers: 2,
  architecture: 'ampere_a6000'
});

assert.equal(defaultPolicy.valid, true);
assert.equal(defaultPolicy.warps, 4);
assert.equal(defaultPolicy.threads, 128);
assert.equal(defaultPolicy.outputPerThread, 32);
assert.equal(defaultPolicy.smemBytes, 10240);
assert.equal(defaultPolicy.mainIntensity, 12.8);
assert.ok(Math.abs(defaultPolicy.fullIntensity - 12.19047619047619) < 1e-12);
assert.equal(defaultPolicy.activeBlocks, 9);
assert.equal(defaultPolicy.occupancy, 0.75);

for (const bk of [8, 16, 32]) {
  const result = evaluate({
    bm: 128,
    bn: 32,
    bk,
    k: 1024,
    warpM: 64,
    warpN: 16,
    threadM: 8,
    threadN: 4,
    buffers: 1
  });
  assert.equal(result.valid, true);
  assert.equal(result.mainIntensity, 12.8, 'A/B main-loop intensity must not depend on BK');
}

const invalidLaneMapping = evaluate({
  bm: 32,
  bn: 16,
  bk: 8,
  warpM: 32,
  warpN: 16,
  threadM: 16,
  threadN: 16
});
assert.equal(invalidLaneMapping.valid, false);
assert.match(invalidLaneMapping.errors.join('\n'), /32 个 Thread tile/);

const invalidBlockPartition = evaluate({
  bm: 96,
  bn: 64,
  bk: 8,
  warpM: 64,
  warpN: 64,
  threadM: 8,
  threadN: 16
});
assert.equal(invalidBlockPartition.valid, false);
assert.match(invalidBlockPartition.errors.join('\n'), /Block tile/);

console.log('model.test.js: all assertions passed');

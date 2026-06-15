'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');

const sgemm = read('sgemm-practice/index.html');
assert.doesNotMatch(sgemm, /reuse:\s*\[/);
assert.match(sgemm, /不是实测复用率/);
assert.match(sgemm, /mechanisms:\[/);

const cutlass = read('cutlass-hierarchy/index.html');
assert.match(cutlass, /assets\/gemm-policy\.js/);
assert.match(cutlass, /A\/B 主循环强度/);
assert.match(cutlass, /threadM:v\.tm,threadN:v\.tn/);
assert.doesNotMatch(cutlass, /Math\.ceil\(outputs\/perThread\)/);

const step5 = read('demos/step5-warp-tiling-cutlass.html');
assert.match(step5, /assets\/gemm-policy\.js/);
assert.match(step5, /Ampere A6000 示例/);
assert.match(step5, /#8B5CF6/);

const step4 = read('demos/step4-vectorized-memory-access.html');
assert.match(step4, /不等价于 profiler 报告的实际硬件内存事务数/);

for (const page of [
  'index.html',
  'cutlass-hierarchy/index.html',
  'sgemm-practice/index.html',
  'cuda-matmul-worklog/index.html',
  'demos/preface-memory-hierarchy.html',
  'demos/step1-naive-sgemm.html',
  'demos/step2-global-memory-coalescing.html',
  'demos/step3-shared-memory-block-tiling.html',
  'demos/step4-vectorized-memory-access.html',
  'demos/step5-warp-tiling-cutlass.html'
]) {
  assert.match(read(page), /semantic-colors\.css/, `semantic color stylesheet missing: ${page}`);
}

const manifest = JSON.parse(read('doc/source-manifest.json'));
for (const item of manifest.files) {
  const content = fs.readFileSync(path.join(root, item.path));
  const digest = crypto.createHash('sha256').update(content).digest('hex');
  assert.equal(digest, item.sha256, `source checksum mismatch: ${item.path}`);
}

console.log('static.test.js: all assertions passed');

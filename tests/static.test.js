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
assert.match(sgemm, /SGEMM Kernel 执行轨迹/);
assert.match(sgemm, /mechanisms:\s*\[/);
assert.match(sgemm, /threadIdx\.x/);
assert.match(sgemm, /FETCH_FLOAT4/);
assert.match(sgemm, /双缓存预取/);

const cutlass = read('cutlass-hierarchy/index.html');
assert.match(cutlass, /assets\/gemm-policy\.js/);
assert.match(cutlass, /A\/B 主循环强度/);
assert.match(cutlass, /threadM:v\.tm,threadN:v\.tn/);
assert.doesNotMatch(cutlass, /Math\.ceil\(outputs\/perThread\)/);

const step5 = read('demos/step5-warp-tiling-cutlass.html');
assert.match(step5, /assets\/gemm-policy\.js/);
assert.match(step5, /Ampere A6000 示例/);
assert.match(step5, /#8B5CF6/);
assert.match(step5, /资源与流水/);
assert.match(step5, /id="prev"/);
assert.match(step5, /id="next"/);

const step4 = read('demos/step4-vectorized-memory-access.html');
assert.match(step4, /不等价于 profiler 报告的实际硬件内存事务数/);
assert.match(step4, /标量基线/);
assert.match(step4, /id="prev"/);
assert.match(step4, /id="next"/);

const step3 = read('demos/step3-shared-memory-block-tiling.html');
assert.match(step3, /id="prev"/);
assert.match(step3, /id="next"/);
assert.match(step3, /phase===4&&last/);

for (const demo of [
  'demos/preface-memory-hierarchy.html',
  'demos/step1-naive-sgemm.html',
  'demos/step2-global-memory-coalescing.html',
  'demos/step3-shared-memory-block-tiling.html',
  'demos/step4-vectorized-memory-access.html',
  'demos/step5-warp-tiling-cutlass.html'
]) {
  assert.match(read(demo), /https:\/\/(siboehm\.com\/articles\/22\/CUDA-MMM|github\.com\/wangzyon\/NVIDIA_SGEMM_PRACTICE|developer\.nvidia\.com\/blog\/cutlass-linear-algebra-cuda\/)/, `source URL missing: ${demo}`);
  assert.doesNotMatch(read(demo), /href="\.\.\/(cuda-matmul-worklog|sgemm-practice|cutlass-hierarchy)\/original\.md"/, `local original link should not be used: ${demo}`);
}

assert.match(read('index.html'), /步骤与原文索引/);
assert.match(read('index.html'), /https:\/\/developer\.nvidia\.com\/blog\/cutlass-linear-algebra-cuda\//);
assert.match(read('index.html'), /https:\/\/github\.com\/wangzyon\/NVIDIA_SGEMM_PRACTICE/);
assert.match(read('index.html'), /https:\/\/siboehm\.com\/articles\/22\/CUDA-MMM/);
assert.doesNotMatch(read('index.html'), /href="\.\/(cuda-matmul-worklog|sgemm-practice|cutlass-hierarchy)\/original\.md"/);

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

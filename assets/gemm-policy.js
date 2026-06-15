(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.GemmPolicy = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const ARCHITECTURES = {
    ampere_a6000: {
      label: 'Ampere A6000 示例',
      smemPerSm: 102400,
      defaultSmemPerBlock: 49152,
      registersPerSm: 65536,
      maxThreadsPerSm: 1536,
      maxThreadsPerBlock: 1024,
      maxBlocksPerSm: 32,
      maxRegistersPerThread: 255
    },
    ampere_generic: {
      label: 'Ampere 通用估算',
      smemPerSm: 102400,
      defaultSmemPerBlock: 49152,
      registersPerSm: 65536,
      maxThreadsPerSm: 2048,
      maxThreadsPerBlock: 1024,
      maxBlocksPerSm: 32,
      maxRegistersPerThread: 255
    }
  };

  function evaluate(config) {
    const v = Object.assign({
      bm: 128,
      bn: 128,
      bk: 8,
      k: 1024,
      warpM: 64,
      warpN: 64,
      threadM: 8,
      threadN: 16,
      buffers: 2,
      architecture: 'ampere_a6000'
    }, config);
    const arch = ARCHITECTURES[v.architecture] || ARCHITECTURES.ampere_a6000;
    const errors = [];
    const warnings = [];

    const positiveKeys = ['bm', 'bn', 'bk', 'k', 'warpM', 'warpN', 'threadM', 'threadN', 'buffers'];
    positiveKeys.forEach(key => {
      if (!Number.isFinite(v[key]) || v[key] <= 0) errors.push(`${key} 必须为正数`);
    });
    if (errors.length) return { valid: false, errors, warnings, architecture: arch };

    if (v.bm % v.warpM || v.bn % v.warpN) errors.push('Block tile 必须能被 Warp tile 整除');
    if (v.warpM % v.threadM || v.warpN % v.threadN) errors.push('Warp tile 必须能被 Thread tile 整除');

    const warpRows = v.bm / v.warpM;
    const warpCols = v.bn / v.warpN;
    const warps = warpRows * warpCols;
    const threads = warps * 32;
    const threadTilesPerWarp = (v.warpM / v.threadM) * (v.warpN / v.threadN);
    if (threadTilesPerWarp !== 32) {
      errors.push(`每个 Warp 必须恰好映射 32 个 Thread tile，当前为 ${threadTilesPerWarp}`);
    }
    if (threads > arch.maxThreadsPerBlock) {
      errors.push(`线程数 ${threads} 超过 ${arch.maxThreadsPerBlock}/block`);
    }

    const outputs = v.bm * v.bn;
    const outputPerThread = v.threadM * v.threadN;
    const inputFloatsPerKTile = (v.bm + v.bn) * v.bk;
    const smemBytes = inputFloatsPerKTile * 4 * v.buffers;
    const mainFlops = 2 * v.bm * v.bn * v.bk;
    const mainBytes = inputFloatsPerKTile * 4;
    const fullFlops = 2 * v.bm * v.bn * v.k;
    const fullBytes = (v.bm * v.k + v.bn * v.k + 2 * v.bm * v.bn) * 4;
    const registerLowerBound = outputPerThread + v.buffers * (v.threadM + v.threadN);

    if (smemBytes > arch.defaultSmemPerBlock) {
      warnings.push(`SMEM ${formatBytes(smemBytes)} 超过默认 ${formatBytes(arch.defaultSmemPerBlock)}/block`);
    }
    if (registerLowerBound > arch.maxRegistersPerThread) {
      errors.push(`寄存器工作集下界 ${registerLowerBound} 超过 ${arch.maxRegistersPerThread}/thread`);
    }

    const smemBlocks = Math.floor(arch.smemPerSm / smemBytes);
    const threadBlocks = Math.floor(arch.maxThreadsPerSm / threads);
    const registerBlocks = Math.floor(arch.registersPerSm / (registerLowerBound * threads));
    const activeBlocks = errors.length ? 0 : Math.max(
      0,
      Math.min(arch.maxBlocksPerSm, smemBlocks, threadBlocks, registerBlocks)
    );
    const occupancy = errors.length ? 0 : Math.min(1, activeBlocks * threads / arch.maxThreadsPerSm);
    if (!errors.length && activeBlocks === 0) errors.push('资源模型无法驻留一个 block');

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      architecture: arch,
      values: v,
      warpRows,
      warpCols,
      warps,
      threads,
      outputs,
      outputPerThread,
      inputFloatsPerKTile,
      smemBytes,
      mainIntensity: mainFlops / mainBytes,
      fullIntensity: fullFlops / fullBytes,
      registerLowerBound,
      activeBlocks,
      occupancy
    };
  }

  function formatBytes(bytes) {
    return bytes >= 1024 ? `${(bytes / 1024).toFixed(1)} KiB` : `${bytes} B`;
  }

  return { ARCHITECTURES, evaluate, formatBytes };
}));

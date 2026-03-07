import { Matrix } from "../matrix/Matrix.js";
import * as cpu from "../matrix/ops.js";
import {
  MATMUL_SHADER,
  ELEMENTWISE_SHADER,
  ACTIVATION_SHADER,
  ADD_BIAS_SHADER,
} from "./shaders.js";

/** Activation function identifiers used in GPU shaders. */
export type ActivationFn = "relu" | "sigmoid" | "tanh";

/** Element-wise binary operation identifiers used in GPU shaders. */
type ElementwiseOp = "add" | "sub" | "hadamard";

const ELEMENTWISE_OP_ID: Record<ElementwiseOp, number> = {
  add: 0,
  sub: 1,
  hadamard: 2,
};

const ACTIVATION_FN_ID: Record<ActivationFn, number> = {
  relu: 0,
  sigmoid: 1,
  tanh: 2,
};

/**
 * Thin wrapper around a WebGPU device that exposes GPU-accelerated matrix
 * operations with automatic CPU fallback when WebGPU is unavailable.
 */
export class GPUContext {
  private device: GPUDevice | null;

  private constructor(device: GPUDevice | null) {
    this.device = device;
  }

  /** Whether a real GPU device was acquired. */
  get isGPU(): boolean {
    return this.device !== null;
  }

  /**
   * Acquire a WebGPU device.
   * Returns a context backed by a real GPU if WebGPU is available in the
   * current environment; otherwise falls back to CPU-only mode.
   */
  static async create(): Promise<GPUContext> {
    if (typeof navigator !== "undefined" && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const device = await adapter.requestDevice();
          return new GPUContext(device);
        }
      } catch {
        // fall through to CPU
      }
    }
    return new GPUContext(null);
  }

  /** Create a CPU-only context without attempting GPU initialisation. */
  static createCPU(): GPUContext {
    return new GPUContext(null);
  }

  // ── Public matrix operations ──────────────────────────────────────────────

  /** C = A + B (element-wise) */
  async add(a: Matrix, b: Matrix): Promise<Matrix> {
    return this.device
      ? this.elementwise(a, b, "add")
      : cpu.add(a, b);
  }

  /** C = A - B (element-wise) */
  async sub(a: Matrix, b: Matrix): Promise<Matrix> {
    return this.device
      ? this.elementwise(a, b, "sub")
      : cpu.sub(a, b);
  }

  /** C = A ⊙ B (element-wise / Hadamard product) */
  async hadamard(a: Matrix, b: Matrix): Promise<Matrix> {
    return this.device
      ? this.elementwise(a, b, "hadamard")
      : cpu.hadamard(a, b);
  }

  /** C = A × B (matrix multiplication) */
  async matmul(a: Matrix, b: Matrix): Promise<Matrix> {
    return this.device
      ? this.gpuMatmul(a, b)
      : cpu.matmul(a, b);
  }

  /**
   * Add bias vector to every row.
   * A is (M×N), bias is (1×N), result is (M×N).
   */
  async addBias(a: Matrix, bias: Matrix): Promise<Matrix> {
    return this.device
      ? this.gpuAddBias(a, bias)
      : cpu.addBias(a, bias);
  }

  /** Apply an activation function element-wise. */
  async activate(a: Matrix, fn: ActivationFn): Promise<Matrix> {
    return this.device
      ? this.gpuActivate(a, fn)
      : applyActivationCPU(a, fn);
  }

  /** Release the underlying WebGPU device. */
  destroy(): void {
    this.device?.destroy();
    this.device = null;
  }

  // ── Private GPU helpers ───────────────────────────────────────────────────

  private async elementwise(
    a: Matrix,
    b: Matrix,
    op: ElementwiseOp,
  ): Promise<Matrix> {
    const device = this.device!;
    const len = a.data.length;

    const [bufA, bufB, bufC] = createStorageBuffers(device, [a.data, b.data, null], len);

    const uniformData = new Uint32Array([len, ELEMENTWISE_OP_ID[op]]);
    const bufUniform = createUniformBuffer(device, uniformData.buffer);

    const pipeline = await createComputePipeline(device, ELEMENTWISE_SHADER);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufUniform } },
      ],
    });

    const workgroups = Math.ceil(len / 64);
    await runComputePass(device, pipeline, bindGroup, workgroups, 1, 1);

    const result = await readBuffer(device, bufC, len);
    destroyBuffers(bufA, bufB, bufC, bufUniform);
    return new Matrix(a.rows, a.cols, result);
  }

  private async gpuMatmul(a: Matrix, b: Matrix): Promise<Matrix> {
    if (a.cols !== b.rows) {
      throw new RangeError(
        `matmul: incompatible shapes ${a.rows}×${a.cols} and ${b.rows}×${b.cols}`,
      );
    }
    const device = this.device!;
    const M = a.rows, K = a.cols, N = b.cols;

    const [bufA, bufB, bufC] = createStorageBuffers(
      device,
      [a.data, b.data, null],
      M * N,
    );

    const uniformData = new Uint32Array([M, K, N]);
    const bufUniform = createUniformBuffer(device, uniformData.buffer);

    const pipeline = await createComputePipeline(device, MATMUL_SHADER);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufUniform } },
      ],
    });

    const wgX = Math.ceil(N / 8);
    const wgY = Math.ceil(M / 8);
    await runComputePass(device, pipeline, bindGroup, wgX, wgY, 1);

    const result = await readBuffer(device, bufC, M * N);
    destroyBuffers(bufA, bufB, bufC, bufUniform);
    return new Matrix(M, N, result);
  }

  private async gpuAddBias(a: Matrix, bias: Matrix): Promise<Matrix> {
    if (bias.rows !== 1 || bias.cols !== a.cols) {
      throw new RangeError(
        `addBias: bias must be 1×${a.cols}, got ${bias.rows}×${bias.cols}`,
      );
    }
    const device = this.device!;
    const M = a.rows, N = a.cols;
    const len = M * N;

    const [bufA, bufBias, bufC] = createStorageBuffers(
      device,
      [a.data, bias.data, null],
      len,
    );

    const uniformData = new Uint32Array([M, N]);
    const bufUniform = createUniformBuffer(device, uniformData.buffer);

    const pipeline = await createComputePipeline(device, ADD_BIAS_SHADER);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufBias } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufUniform } },
      ],
    });

    await runComputePass(device, pipeline, bindGroup, Math.ceil(len / 64), 1, 1);

    const result = await readBuffer(device, bufC, len);
    destroyBuffers(bufA, bufBias, bufC, bufUniform);
    return new Matrix(M, N, result);
  }

  private async gpuActivate(a: Matrix, fn: ActivationFn): Promise<Matrix> {
    const device = this.device!;
    const len = a.data.length;

    const bufA = createStorageBuffer(device, a.data);
    const bufB = device.createBuffer({
      size: len * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const uniformData = new Uint32Array([len, ACTIVATION_FN_ID[fn]]);
    const bufUniform = createUniformBuffer(device, uniformData.buffer);

    const pipeline = await createComputePipeline(device, ACTIVATION_SHADER);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufUniform } },
      ],
    });

    await runComputePass(device, pipeline, bindGroup, Math.ceil(len / 64), 1, 1);

    const result = await readBuffer(device, bufB, len);
    destroyBuffers(bufA, bufB, bufUniform);
    return new Matrix(a.rows, a.cols, result);
  }
}

// ── WebGPU utility helpers ────────────────────────────────────────────────────

function createStorageBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buf;
}

/**
 * Create multiple storage buffers.
 * `dataSources` is an array where each entry is either a Float32Array (filled
 * immediately) or null (empty output buffer of `outputLen` f32 elements).
 */
function createStorageBuffers(
  device: GPUDevice,
  dataSources: (Float32Array | null)[],
  outputLen: number,
): GPUBuffer[] {
  return dataSources.map((src) => {
    if (src !== null) {
      return createStorageBuffer(device, src);
    }
    return device.createBuffer({
      size: outputLen * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  });
}

function createUniformBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  // Uniform buffers must be aligned to 16 bytes.
  const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
  const buf = device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

async function createComputePipeline(
  device: GPUDevice,
  wgsl: string,
): Promise<GPUComputePipeline> {
  return device.createComputePipelineAsync({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: wgsl }),
      entryPoint: "main",
    },
  });
}

async function runComputePass(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  wgX: number,
  wgY: number,
  wgZ: number,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY, wgZ);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function readBuffer(
  device: GPUDevice,
  srcBuf: GPUBuffer,
  len: number,
): Promise<Float32Array> {
  const stagingBuf = device.createBuffer({
    size: len * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(srcBuf, 0, stagingBuf, 0, len * 4);
  device.queue.submit([encoder.finish()]);
  await stagingBuf.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();
  stagingBuf.destroy();
  return result;
}

function destroyBuffers(...bufs: GPUBuffer[]): void {
  for (const buf of bufs) buf.destroy();
}

// ── CPU activation helpers ────────────────────────────────────────────────────

function applyActivationCPU(a: Matrix, fn: ActivationFn): Matrix {
  switch (fn) {
    case "relu":    return cpu.relu(a);
    case "sigmoid": return cpu.sigmoid(a);
    case "tanh":    return cpu.tanh(a);
  }
}

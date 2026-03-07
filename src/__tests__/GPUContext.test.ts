import { describe, it, expect } from "vitest";
import { Matrix } from "../matrix/Matrix.js";
import { GPUContext } from "../gpu/GPUContext.js";

/**
 * All tests run through GPUContext in CPU-fallback mode (no WebGPU in test env).
 * This validates that the public API, routing logic, and CPU fallback all work.
 */

async function cpuCtx(): Promise<GPUContext> {
  return GPUContext.createCPU();
}

const close = (a: number, b: number, eps = 1e-5) => Math.abs(a - b) < eps;

describe("GPUContext (CPU fallback)", () => {
  it("reports isGPU = false", async () => {
    const ctx = await cpuCtx();
    expect(ctx.isGPU).toBe(false);
  });

  it("add", async () => {
    const ctx = await cpuCtx();
    const a = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = Matrix.fromArray([[5, 6], [7, 8]]);
    const c = await ctx.add(a, b);
    expect(c.toArray()).toEqual([[6, 8], [10, 12]]);
  });

  it("sub", async () => {
    const ctx = await cpuCtx();
    const a = Matrix.fromArray([[10, 20]]);
    const b = Matrix.fromArray([[3, 7]]);
    const c = await ctx.sub(a, b);
    expect(c.toArray()).toEqual([[7, 13]]);
  });

  it("hadamard", async () => {
    const ctx = await cpuCtx();
    const a = Matrix.fromArray([[2, 3], [4, 5]]);
    const b = Matrix.fromArray([[1, 2], [3, 4]]);
    const c = await ctx.hadamard(a, b);
    expect(c.toArray()).toEqual([[2, 6], [12, 20]]);
  });

  it("matmul", async () => {
    const ctx = await cpuCtx();
    const a = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = Matrix.fromArray([[5, 6], [7, 8]]);
    const c = await ctx.matmul(a, b);
    expect(c.toArray()).toEqual([[19, 22], [43, 50]]);
  });

  it("addBias", async () => {
    const ctx = await cpuCtx();
    const a = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
    const bias = Matrix.fromArray([[10, 20, 30]]);
    const c = await ctx.addBias(a, bias);
    expect(c.toArray()).toEqual([[11, 22, 33], [14, 25, 36]]);
  });

  it("activate relu", async () => {
    const ctx = await cpuCtx();
    const a = Matrix.fromArray([[-1, 0, 1, 2]]);
    const c = await ctx.activate(a, "relu");
    expect(c.toArray()).toEqual([[0, 0, 1, 2]]);
  });

  it("activate sigmoid", async () => {
    const ctx = await cpuCtx();
    const a = new Matrix(1, 1, [0]);
    const c = await ctx.activate(a, "sigmoid");
    expect(close(c.get(0, 0), 0.5)).toBe(true);
  });

  it("activate tanh", async () => {
    const ctx = await cpuCtx();
    const a = new Matrix(1, 1, [0]);
    const c = await ctx.activate(a, "tanh");
    expect(close(c.get(0, 0), 0)).toBe(true);
  });
});

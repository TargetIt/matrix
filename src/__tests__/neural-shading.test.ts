import { describe, it, expect } from "vitest";
import { Matrix } from "../matrix/Matrix.js";
import { GPUContext } from "../gpu/GPUContext.js";
import { LinearLayer, MLP } from "../neural-shading/MLP.js";
import { NeuralShader } from "../neural-shading/NeuralShader.js";

async function cpuCtx(): Promise<GPUContext> {
  return GPUContext.createCPU();
}

const allInRange = (m: Matrix, lo: number, hi: number) => {
  for (const v of m.data) {
    if (v < lo || v > hi) return false;
  }
  return true;
};

describe("LinearLayer", () => {
  it("forward produces correct output shape", async () => {
    const ctx = await cpuCtx();
    const layer = new LinearLayer(4, 8, "relu");
    const x = Matrix.random(3, 4); // batch=3, inFeatures=4
    const y = await layer.forward(x, ctx);
    expect(y.rows).toBe(3);
    expect(y.cols).toBe(8);
  });

  it("forward with relu has non-negative outputs", async () => {
    const ctx = await cpuCtx();
    const layer = new LinearLayer(4, 8, "relu");
    const x = Matrix.random(5, 4, -1, 1);
    const y = await layer.forward(x, ctx);
    expect(allInRange(y, 0, Infinity)).toBe(true);
  });

  it("forward with sigmoid has outputs in (0, 1)", async () => {
    const ctx = await cpuCtx();
    const layer = new LinearLayer(4, 8, "sigmoid");
    const x = Matrix.random(5, 4, -10, 10);
    const y = await layer.forward(x, ctx);
    expect(allInRange(y, 0, 1)).toBe(true);
  });

  it("setParameters validates shapes", () => {
    const layer = new LinearLayer(4, 8);
    expect(() =>
      layer.setParameters(Matrix.zeros(3, 4), Matrix.zeros(1, 8)),
    ).toThrow(RangeError);
    expect(() =>
      layer.setParameters(Matrix.zeros(8, 4), Matrix.zeros(1, 3)),
    ).toThrow(RangeError);
  });

  it("setParameters accepts correct shapes", () => {
    const layer = new LinearLayer(4, 8);
    const w = Matrix.zeros(8, 4);
    const b = Matrix.zeros(1, 8);
    expect(() => layer.setParameters(w, b)).not.toThrow();
  });
});

describe("MLP", () => {
  it("forward propagates through layers", async () => {
    const ctx = await cpuCtx();
    const layers = [
      new LinearLayer(4, 16, "relu"),
      new LinearLayer(16, 8, "relu"),
      new LinearLayer(8, 3, "sigmoid"),
    ];
    const mlp = new MLP(layers, ctx);
    const x = Matrix.random(2, 4);
    const y = await mlp.forward(x);
    expect(y.rows).toBe(2);
    expect(y.cols).toBe(3);
    // sigmoid output ∈ (0, 1)
    expect(allInRange(y, 0, 1)).toBe(true);
  });

  it("single layer MLP works", async () => {
    const ctx = await cpuCtx();
    const mlp = new MLP([new LinearLayer(2, 4, "relu")], ctx);
    const x = Matrix.random(1, 2);
    const y = await mlp.forward(x);
    expect(y.rows).toBe(1);
    expect(y.cols).toBe(4);
  });
});

describe("NeuralShader", () => {
  it("shades a single point and returns RGB", async () => {
    const ctx = await cpuCtx();
    const shader = NeuralShader.create(ctx);
    const rgb = await shader.shadePoint({
      position: [0, 1, 0],
      normal:   [0, 1, 0],
      viewDir:  [0, -1, 0],
    });
    expect(typeof rgb.r).toBe("number");
    expect(typeof rgb.g).toBe("number");
    expect(typeof rgb.b).toBe("number");
    // sigmoid output is in (0, 1)
    expect(rgb.r).toBeGreaterThanOrEqual(0);
    expect(rgb.r).toBeLessThanOrEqual(1);
    expect(rgb.g).toBeGreaterThanOrEqual(0);
    expect(rgb.g).toBeLessThanOrEqual(1);
    expect(rgb.b).toBeGreaterThanOrEqual(0);
    expect(rgb.b).toBeLessThanOrEqual(1);
  });

  it("shades a batch of points", async () => {
    const ctx = await cpuCtx();
    const shader = NeuralShader.create(ctx);
    const inputs = [
      { position: [0, 0, 0] as [number,number,number], normal: [0, 1, 0] as [number,number,number], viewDir: [0, -1, 0] as [number,number,number] },
      { position: [1, 0, 0] as [number,number,number], normal: [1, 0, 0] as [number,number,number], viewDir: [-1, 0, 0] as [number,number,number] },
      { position: [0, 0, 1] as [number,number,number], normal: [0, 0, 1] as [number,number,number], viewDir: [0, 0, -1] as [number,number,number] },
    ];
    const results = await shader.shade(inputs);
    expect(results).toHaveLength(3);
    for (const rgb of results) {
      expect(rgb.r).toBeGreaterThanOrEqual(0);
      expect(rgb.r).toBeLessThanOrEqual(1);
    }
  });

  it("returns empty array for empty input", async () => {
    const ctx = await cpuCtx();
    const shader = NeuralShader.create(ctx);
    const results = await shader.shade([]);
    expect(results).toEqual([]);
  });

  it("creates with custom config", async () => {
    const ctx = await cpuCtx();
    const shader = NeuralShader.create(ctx, {
      hiddenSize: 32,
      hiddenLayers: 2,
      hiddenActivation: "tanh",
    });
    const rgb = await shader.shadePoint({
      position: [0, 0, 0],
      normal:   [0, 1, 0],
      viewDir:  [0, -1, 0],
    });
    expect(rgb.r).toBeGreaterThanOrEqual(0);
    expect(rgb.r).toBeLessThanOrEqual(1);
  });

  it("exposes network layers", async () => {
    const ctx = await cpuCtx();
    const shader = NeuralShader.create(ctx, { hiddenLayers: 2 });
    // 2 hidden + 1 output = 3 layers total
    expect(shader.network.layers).toHaveLength(3);
  });
});

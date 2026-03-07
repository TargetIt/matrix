import { Matrix } from "../matrix/Matrix.js";
import { GPUContext, ActivationFn } from "../gpu/GPUContext.js";
import * as cpu from "../matrix/ops.js";

/**
 * A single fully-connected (linear) layer with an optional activation.
 *
 * Stores weights W (outFeatures × inFeatures) and bias b (1 × outFeatures).
 */
export class LinearLayer {
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly activation: ActivationFn | null;
  weights: Matrix; // shape: outFeatures × inFeatures
  bias: Matrix;    // shape: 1 × outFeatures

  constructor(
    inFeatures: number,
    outFeatures: number,
    activation: ActivationFn | null = null,
  ) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.activation = activation;
    // He / Glorot-like uniform initialisation
    const limit = Math.sqrt(6 / (inFeatures + outFeatures));
    this.weights = Matrix.random(outFeatures, inFeatures, -limit, limit);
    this.bias = Matrix.zeros(1, outFeatures);
  }

  /**
   * Forward pass: given input X (batchSize × inFeatures),
   * compute  Y = activate(X · Wᵀ + b).
   * Returns Y (batchSize × outFeatures).
   */
  async forward(x: Matrix, ctx: GPUContext): Promise<Matrix> {
    // X  : batchSize × inFeatures
    // Wᵀ : inFeatures × outFeatures
    const wt = cpu.transpose(this.weights);
    let out = await ctx.matmul(x, wt);
    out = await ctx.addBias(out, this.bias);
    if (this.activation !== null) {
      out = await ctx.activate(out, this.activation);
    }
    return out;
  }

  /** Load pre-trained weights and bias (e.g. from a serialised model). */
  setParameters(weights: Matrix, bias: Matrix): void {
    if (
      weights.rows !== this.outFeatures ||
      weights.cols !== this.inFeatures
    ) {
      throw new RangeError(
        `setParameters: expected weights ${this.outFeatures}×${this.inFeatures}, ` +
        `got ${weights.rows}×${weights.cols}`,
      );
    }
    if (bias.rows !== 1 || bias.cols !== this.outFeatures) {
      throw new RangeError(
        `setParameters: expected bias 1×${this.outFeatures}, ` +
        `got ${bias.rows}×${bias.cols}`,
      );
    }
    this.weights = weights;
    this.bias = bias;
  }
}

/**
 * A multi-layer perceptron (MLP) composed of LinearLayer instances.
 */
export class MLP {
  readonly layers: LinearLayer[];
  private ctx: GPUContext;

  constructor(layers: LinearLayer[], ctx: GPUContext) {
    this.layers = layers;
    this.ctx = ctx;
  }

  /** Run a full forward pass through all layers. */
  async forward(x: Matrix): Promise<Matrix> {
    let out = x;
    for (const layer of this.layers) {
      out = await layer.forward(out, this.ctx);
    }
    return out;
  }
}

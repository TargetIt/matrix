import { Matrix } from "../matrix/Matrix.js";
import { GPUContext } from "../gpu/GPUContext.js";
import { LinearLayer, MLP } from "./MLP.js";

/**
 * Shading input features for a single surface point.
 *
 * Typical inputs for a neural shading network:
 *   - position (x, y, z)
 *   - surface normal (nx, ny, nz)
 *   - view direction (vx, vy, vz)
 *   - any additional encoded features (e.g. positional encoding)
 */
export interface ShadingInput {
  /** World-space position [x, y, z]. */
  position: [number, number, number];
  /** Unit surface normal [nx, ny, nz]. */
  normal: [number, number, number];
  /** Unit view direction [vx, vy, vz]. */
  viewDir: [number, number, number];
}

/** Shading output: an RGB colour in [0, 1]. */
export interface ShadingOutput {
  r: number;
  g: number;
  b: number;
}

/**
 * Configuration for building a NeuralShader.
 */
export interface NeuralShaderConfig {
  /** Number of input features (default: 9 for position + normal + viewDir). */
  inputFeatures?: number;
  /** Number of hidden units per layer (default: 64). */
  hiddenSize?: number;
  /** Number of hidden layers (default: 3). */
  hiddenLayers?: number;
  /** Activation function for hidden layers (default: "relu"). */
  hiddenActivation?: "relu" | "sigmoid" | "tanh";
}

/**
 * Neural Shader: a small MLP that maps surface shading inputs to RGB colour.
 *
 * Architecture:
 *   input → [Linear+ReLU] × hiddenLayers → Linear+Sigmoid → RGB output
 *
 * The final sigmoid activation bounds output values to (0, 1), which is
 * suitable for HDR-normalised radiance estimates.
 */
export class NeuralShader {
  private mlp: MLP;
  private inputFeatures: number;

  private constructor(mlp: MLP, inputFeatures: number) {
    this.mlp = mlp;
    this.inputFeatures = inputFeatures;
  }

  /**
   * Build a new NeuralShader with randomly-initialised weights.
   *
   * @param ctx     - GPU context to use for inference.
   * @param config  - Architecture hyperparameters.
   */
  static create(ctx: GPUContext, config: NeuralShaderConfig = {}): NeuralShader {
    const inFeatures = config.inputFeatures ?? 9;
    const hidden = config.hiddenSize ?? 64;
    const depth = config.hiddenLayers ?? 3;
    const hiddenAct = config.hiddenActivation ?? "relu";

    const layers: LinearLayer[] = [];

    // First layer: inFeatures → hidden
    layers.push(new LinearLayer(inFeatures, hidden, hiddenAct));

    // Intermediate hidden layers
    for (let i = 1; i < depth; i++) {
      layers.push(new LinearLayer(hidden, hidden, hiddenAct));
    }

    // Output layer: hidden → 3 (RGB), bounded by sigmoid
    layers.push(new LinearLayer(hidden, 3, "sigmoid"));

    return new NeuralShader(new MLP(layers, ctx), inFeatures);
  }

  /**
   * Run neural shading inference on a batch of surface points.
   *
   * @param inputs  - Array of shading inputs.
   * @returns       - Array of RGB colour outputs (same length as inputs).
   */
  async shade(inputs: ShadingInput[]): Promise<ShadingOutput[]> {
    if (inputs.length === 0) return [];

    const featureMatrix = this.encodeInputs(inputs);
    const output = await this.mlp.forward(featureMatrix);

    const results: ShadingOutput[] = [];
    for (let i = 0; i < inputs.length; i++) {
      results.push({
        r: output.get(i, 0),
        g: output.get(i, 1),
        b: output.get(i, 2),
      });
    }
    return results;
  }

  /**
   * Shade a single surface point (convenience wrapper).
   */
  async shadePoint(input: ShadingInput): Promise<ShadingOutput> {
    const [result] = await this.shade([input]);
    return result;
  }

  /**
   * Encode an array of ShadingInputs into a (batchSize × inputFeatures) Matrix.
   * Default encoding: [px, py, pz, nx, ny, nz, vx, vy, vz].
   */
  private encodeInputs(inputs: ShadingInput[]): Matrix {
    const flat: number[] = [];
    for (const inp of inputs) {
      flat.push(
        ...inp.position,
        ...inp.normal,
        ...inp.viewDir,
      );
    }
    // If a custom inputFeatures count was provided and exceeds 9, pad with zeros.
    const baseLen = 9;
    const extra = this.inputFeatures - baseLen;
    if (extra > 0) {
      for (let i = 0; i < inputs.length; i++) {
        for (let j = 0; j < extra; j++) {
          flat.splice((i + 1) * this.inputFeatures - extra + j, 0, 0);
        }
      }
    }
    return new Matrix(inputs.length, this.inputFeatures, flat);
  }

  /** Access the underlying MLP (e.g. to inspect or replace layer weights). */
  get network(): MLP {
    return this.mlp;
  }
}

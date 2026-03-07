/**
 * GPGPU Matrix – GPU-accelerated matrix operations and neural shading.
 *
 * @example
 * ```ts
 * import { Matrix, ops, GPUContext, NeuralShader } from "matrix";
 *
 * // CPU matrix operations
 * const a = Matrix.fromArray([[1, 2], [3, 4]]);
 * const b = Matrix.fromArray([[5, 6], [7, 8]]);
 * const c = ops.matmul(a, b);
 *
 * // GPU-accelerated operations (falls back to CPU when WebGPU unavailable)
 * const ctx = await GPUContext.create();
 * const d = await ctx.matmul(a, b);
 *
 * // Neural shading
 * const shader = NeuralShader.create(ctx);
 * const colour = await shader.shadePoint({
 *   position: [0, 1, 0],
 *   normal:   [0, 1, 0],
 *   viewDir:  [0, -1, 0],
 * });
 * console.log(colour); // { r, g, b }
 * ```
 */

export { Matrix } from "./matrix/Matrix.js";
export * as ops from "./matrix/ops.js";
export { GPUContext } from "./gpu/GPUContext.js";
export type { ActivationFn } from "./gpu/GPUContext.js";
export { LinearLayer, MLP, NeuralShader } from "./neural-shading/index.js";
export type {
  ShadingInput,
  ShadingOutput,
  NeuralShaderConfig,
} from "./neural-shading/index.js";

# matrix

A TypeScript library for **GPGPU matrix operations** and **neural shading**, powered by WebGPU compute shaders with automatic CPU fallback.

## Features

- **GPGPU matrix operations** via WebGPU compute shaders
  - Matrix multiplication (`matmul`)
  - Element-wise add, sub, Hadamard product
  - Bias addition, transpose, scale
  - Activation functions: ReLU, Sigmoid, Tanh, Softmax
  - Automatic CPU fallback when WebGPU is unavailable
- **Neural shading** via a configurable MLP (Multi-Layer Perceptron)
  - Maps surface attributes (position, normal, view direction) → RGB colour
  - Fully GPU-accelerated forward pass
  - Configurable depth, width, and activation functions

## Quick Start

```ts
import { Matrix, ops, GPUContext, NeuralShader } from "matrix";

// ── CPU matrix operations ──────────────────────────────────────────────────
const a = Matrix.fromArray([[1, 2], [3, 4]]);
const b = Matrix.fromArray([[5, 6], [7, 8]]);
const c = ops.matmul(a, b);           // [[19, 22], [43, 50]]
const d = ops.relu(ops.sub(a, b));    // element-wise ops + activation

// ── GPU-accelerated operations ─────────────────────────────────────────────
// Falls back to CPU automatically if WebGPU is not available.
const ctx = await GPUContext.create();
const e = await ctx.matmul(a, b);     // same result, GPU path when available

// ── Neural shading ─────────────────────────────────────────────────────────
const shader = NeuralShader.create(ctx, {
  hiddenSize: 64,
  hiddenLayers: 3,
  hiddenActivation: "relu",
});

const colour = await shader.shadePoint({
  position: [0, 1, 0],
  normal:   [0, 1, 0],
  viewDir:  [0, -1, 0],
});
// colour = { r: number, g: number, b: number }  — each in [0, 1]

// Batch shading
const colours = await shader.shade([
  { position: [0, 0, 0], normal: [0, 1, 0], viewDir: [0, -1, 0] },
  { position: [1, 0, 0], normal: [1, 0, 0], viewDir: [-1, 0, 0] },
]);
```

## API

### `Matrix`

| Method / static | Description |
|---|---|
| `new Matrix(rows, cols, data?)` | Construct from optional flat `Float32Array` or `number[]` |
| `Matrix.zeros(r, c)` | r×c zero matrix |
| `Matrix.identity(n)` | n×n identity matrix |
| `Matrix.random(r, c, min, max)` | Uniform random matrix |
| `Matrix.fromArray(values)` | From nested array |
| `m.get(r, c)` / `m.set(r, c, v)` | Element access |
| `m.clone()` | Deep copy |
| `m.fill(v)` | Fill with scalar |
| `m.toArray()` | Convert to `number[][]` |

### CPU operations (`ops.*`)

`add`, `sub`, `scale`, `hadamard`, `matmul`, `transpose`, `addBias`, `relu`, `sigmoid`, `tanh`, `softmax`

### `GPUContext`

| Method | Description |
|---|---|
| `GPUContext.create()` | Acquire GPU device; falls back to CPU |
| `GPUContext.createCPU()` | Explicit CPU-only context |
| `ctx.matmul(a, b)` | GPU matrix multiply |
| `ctx.add(a, b)` | GPU element-wise add |
| `ctx.sub(a, b)` | GPU element-wise subtract |
| `ctx.hadamard(a, b)` | GPU Hadamard product |
| `ctx.addBias(a, bias)` | GPU bias addition |
| `ctx.activate(a, fn)` | GPU activation (`"relu"`, `"sigmoid"`, `"tanh"`) |
| `ctx.destroy()` | Release GPU device |

### `NeuralShader`

| Method | Description |
|---|---|
| `NeuralShader.create(ctx, config?)` | Build shader with random weights |
| `shader.shadePoint(input)` | Shade one surface point → `{r, g, b}` |
| `shader.shade(inputs[])` | Batch shading |
| `shader.network` | Access underlying `MLP` |

#### `NeuralShaderConfig`

```ts
interface NeuralShaderConfig {
  inputFeatures?: number;        // default 9, must be >= 9
  hiddenSize?: number;           // default 64
  hiddenLayers?: number;         // default 3, may be 0 for direct input→RGB
  hiddenActivation?: "relu" | "sigmoid" | "tanh";  // default "relu"
}
```

## Run Locally

```bash
npm install
npm run dev     # start dev server, then open http://localhost:5173
```

Open the printed URL in a browser to see an interactive demo of matrix operations and neural shading.

## Development

```bash
npm install
npm run dev     # start dev server with live reload (opens index.html)
npm run build   # compile TypeScript → dist/
npm test        # run vitest test suite
npm run lint    # type-check only (tsc --noEmit)
```

## Architecture

```
src/
├── matrix/
│   ├── Matrix.ts        # Core Matrix class (row-major Float32)
│   └── ops.ts           # CPU implementations of all operations
├── gpu/
│   ├── shaders.ts       # WGSL compute shaders (matmul, elementwise, activation, bias)
│   └── GPUContext.ts    # WebGPU device wrapper + CPU fallback routing
└── neural-shading/
    ├── MLP.ts           # LinearLayer and MLP classes
    └── NeuralShader.ts  # High-level neural shading pipeline
```

import { Matrix } from "./Matrix.js";

/**
 * CPU fallback implementations of all GPGPU matrix operations.
 * These are used when WebGPU is unavailable and directly by tests.
 */

/** Element-wise addition: C = A + B */
export function add(a: Matrix, b: Matrix): Matrix {
  assertSameShape(a, b, "add");
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = a.data[i] + b.data[i];
  }
  return out;
}

/** Element-wise subtraction: C = A - B */
export function sub(a: Matrix, b: Matrix): Matrix {
  assertSameShape(a, b, "sub");
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = a.data[i] - b.data[i];
  }
  return out;
}

/** Scalar multiplication: B = s * A */
export function scale(a: Matrix, s: number): Matrix {
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = a.data[i] * s;
  }
  return out;
}

/** Element-wise (Hadamard) product: C = A ⊙ B */
export function hadamard(a: Matrix, b: Matrix): Matrix {
  assertSameShape(a, b, "hadamard");
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = a.data[i] * b.data[i];
  }
  return out;
}

/**
 * Matrix multiplication: C = A × B
 * A is (M×K), B is (K×N), C is (M×N).
 */
export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new RangeError(
      `matmul: incompatible shapes ${a.rows}×${a.cols} and ${b.rows}×${b.cols}`,
    );
  }
  const M = a.rows;
  const K = a.cols;
  const N = b.cols;
  const out = new Matrix(M, N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a.data[m * K + k] * b.data[k * N + n];
      }
      out.data[m * N + n] = sum;
    }
  }
  return out;
}

/**
 * Transpose: B = Aᵀ
 * A is (M×N), B is (N×M).
 */
export function transpose(a: Matrix): Matrix {
  const out = new Matrix(a.cols, a.rows);
  for (let r = 0; r < a.rows; r++) {
    for (let c = 0; c < a.cols; c++) {
      out.data[c * a.rows + r] = a.data[r * a.cols + c];
    }
  }
  return out;
}

/**
 * Add a bias vector (row vector of length N) to every row of matrix A (M×N).
 * Returns a new (M×N) matrix.
 */
export function addBias(a: Matrix, bias: Matrix): Matrix {
  if (bias.rows !== 1 || bias.cols !== a.cols) {
    throw new RangeError(
      `addBias: bias must be 1×${a.cols}, got ${bias.rows}×${bias.cols}`,
    );
  }
  const out = new Matrix(a.rows, a.cols);
  for (let r = 0; r < a.rows; r++) {
    for (let c = 0; c < a.cols; c++) {
      out.data[r * a.cols + c] = a.data[r * a.cols + c] + bias.data[c];
    }
  }
  return out;
}

// ── Activation functions ──────────────────────────────────────────────────────

/** Element-wise ReLU: max(0, x) */
export function relu(a: Matrix): Matrix {
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = a.data[i] > 0 ? a.data[i] : 0;
  }
  return out;
}

/** Element-wise sigmoid: 1 / (1 + exp(-x)) */
export function sigmoid(a: Matrix): Matrix {
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = 1 / (1 + Math.exp(-a.data[i]));
  }
  return out;
}

/** Element-wise tanh */
export function tanh(a: Matrix): Matrix {
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = Math.tanh(a.data[i]);
  }
  return out;
}

/**
 * Softmax along each row (treating each row as a probability distribution).
 */
export function softmax(a: Matrix): Matrix {
  const out = new Matrix(a.rows, a.cols);
  for (let r = 0; r < a.rows; r++) {
    const base = r * a.cols;
    // Numerical stability: subtract row max
    let rowMax = -Infinity;
    for (let c = 0; c < a.cols; c++) {
      if (a.data[base + c] > rowMax) rowMax = a.data[base + c];
    }
    let sum = 0;
    for (let c = 0; c < a.cols; c++) {
      const e = Math.exp(a.data[base + c] - rowMax);
      out.data[base + c] = e;
      sum += e;
    }
    for (let c = 0; c < a.cols; c++) {
      out.data[base + c] /= sum;
    }
  }
  return out;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function assertSameShape(a: Matrix, b: Matrix, op: string): void {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new RangeError(
      `${op}: shape mismatch ${a.rows}×${a.cols} vs ${b.rows}×${b.cols}`,
    );
  }
}

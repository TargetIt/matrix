/**
 * WGSL compute shader for GPU matrix multiplication.
 *
 * Dispatched with workgroup size (8, 8, 1).
 * Each invocation computes one element of the output matrix C = A × B.
 *
 * Uniforms layout (3 × u32):
 *   offset 0 : M  (rows of A / C)
 *   offset 4 : K  (cols of A / rows of B)
 *   offset 8 : N  (cols of B / C)
 */
export const MATMUL_SHADER = /* wgsl */ `
struct Dims {
  M : u32,
  K : u32,
  N : u32,
};

@group(0) @binding(0) var<storage, read>       A    : array<f32>;
@group(0) @binding(1) var<storage, read>       B    : array<f32>;
@group(0) @binding(2) var<storage, read_write> C    : array<f32>;
@group(0) @binding(3) var<uniform>             dims : Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= dims.M || col >= dims.N) { return; }

  var sum : f32 = 0.0;
  for (var k : u32 = 0u; k < dims.K; k++) {
    sum += A[row * dims.K + k] * B[k * dims.N + col];
  }
  C[row * dims.N + col] = sum;
}
`;

/**
 * WGSL compute shader for element-wise operations on two same-shape matrices.
 *
 * Dispatched with workgroup size (64, 1, 1).
 * Uniform `op` selects the operation:
 *   0 = add, 1 = sub, 2 = hadamard (element-wise multiply)
 *
 * Uniforms layout (2 × u32):
 *   offset 0 : len  (total element count)
 *   offset 4 : op
 */
export const ELEMENTWISE_SHADER = /* wgsl */ `
struct Params {
  len : u32,
  op  : u32,
};

@group(0) @binding(0) var<storage, read>       A      : array<f32>;
@group(0) @binding(1) var<storage, read>       B      : array<f32>;
@group(0) @binding(2) var<storage, read_write> C      : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  if (params.op == 0u) {
    C[i] = A[i] + B[i];
  } else if (params.op == 1u) {
    C[i] = A[i] - B[i];
  } else {
    C[i] = A[i] * B[i];
  }
}
`;

/**
 * WGSL compute shader for element-wise activation functions.
 *
 * Dispatched with workgroup size (64, 1, 1).
 * Uniform `fn_id` selects the function:
 *   0 = relu, 1 = sigmoid, 2 = tanh
 *
 * Uniforms layout (2 × u32):
 *   offset 0 : len
 *   offset 4 : fn_id
 */
export const ACTIVATION_SHADER = /* wgsl */ `
struct Params {
  len   : u32,
  fn_id : u32,
};

@group(0) @binding(0) var<storage, read>       A      : array<f32>;
@group(0) @binding(1) var<storage, read_write> B      : array<f32>;
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  let x = A[i];
  if (params.fn_id == 0u) {
    B[i] = max(0.0, x);
  } else if (params.fn_id == 1u) {
    B[i] = 1.0 / (1.0 + exp(-x));
  } else {
    B[i] = tanh(x);
  }
}
`;

/**
 * WGSL compute shader for adding a bias vector to every row of a matrix.
 *
 * A is (M×N), bias is (1×N), output is (M×N).
 *
 * Uniforms layout (2 × u32):
 *   offset 0 : M
 *   offset 4 : N
 */
export const ADD_BIAS_SHADER = /* wgsl */ `
struct Dims {
  M : u32,
  N : u32,
};

@group(0) @binding(0) var<storage, read>       A    : array<f32>;
@group(0) @binding(1) var<storage, read>       bias : array<f32>;
@group(0) @binding(2) var<storage, read_write> C    : array<f32>;
@group(0) @binding(3) var<uniform>             dims : Dims;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let total = dims.M * dims.N;
  if (i >= total) { return; }
  let col = i % dims.N;
  C[i] = A[i] + bias[col];
}
`;

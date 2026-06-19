"""
CUTLASS Hierarchical GEMM Simulation (0-9 整数矩阵, seed=1)
============================================================
Verification script for 03-cutlass-hierarchy.md

Parameters:
  M = N = K = 16 (problem size)
  BM = 8, BN = 8, BK = 4 (thread block tile)
  WM = 4, WN = 4 (warp tile)
  TM = 2, TN = 2 (thread tile)

Memory layout: row-major (C-style) for all matrices.

CUTLASS uses column-major for A and row-major for B internally,
but for this demo we use row-major throughout for simplicity.
The key conceptual decomposition is identical.

Memory hierarchy labels:
  全局内存 (GMEM / HBM) — matrices A, B, C originally reside here
  共享内存 (SMEM)       — Thread Block Tile loads data here
  寄存器 (Register File / RF) — Warp Tile and Thread Tile compute here

Thread block grid: 2x2 blocks (C is 16x16, block tile is 8x8)
Warp grid per block: 2x2 warps (block tile 8x8, warp tile 4x4)
Thread grid per warp: 2x2 threads (warp tile 4x4, thread tile 2x2)
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(1)
np.set_printoptions(precision=0, suppress=True, linewidth=200)

# ──────────────────────────────────────────────
# 1. Problem definition
# ──────────────────────────────────────────────
M, N, K = 16, 16, 16
A = np.random.randint(0, 10, (M, K)).astype(np.float32)
B = np.random.randint(0, 10, (K, N)).astype(np.float32)
C_expected = A @ B

# ──────────────────────────────────────────────
# 2. Hierarchy parameters
# ──────────────────────────────────────────────
BM, BN, BK = 8, 8, 4   # Thread Block Tile
WM, WN     = 4, 4       # Warp Tile
TM, TN     = 2, 2       # Thread Tile

# Verify divisibility
assert M % BM == 0 and N % BN == 0 and K % BK == 0
assert BM % WM == 0 and BN % WN == 0
assert WM % TM == 0 and WN % TN == 0

num_block_m = M // BM  # 2
num_block_n = N // BN  # 2
num_warp_m  = BM // WM # 2 warps per block (M direction)
num_warp_n  = BN // WN # 2 warps per block (N direction)
num_thr_m   = WM // TM # 2 threads per warp (M direction)
num_thr_n   = WN // TN # 2 threads per warp (N direction)

# Total threads per block = num_warp_m * num_warp_n * num_thr_m * num_thr_n
# = 2 * 2 * 2 * 2 = 16 logical threads per block (each computing a distinct 2x2 tile)
# In real CUTLASS, blocks are 64-256 threads; this is scaled down for clarity.
threads_per_block = num_warp_m * num_warp_n * num_thr_m * num_thr_n

print("=" * 72)
print("CUTLASS Hierarchical GEMM — Verification Script (0-9 整数矩阵, seed=1)")
print("=" * 72)
print(f"\nProblem:  M={M}, N={N}, K={K}")
print(f"数据类型: int32 randint(0,10) → float32")
print(f"随机种子: 1")
print(f"Block Tile:  BM={BM}, BN={BN}, BK={BK}")
print(f"Warp Tile:   WM={WM}, WN={WN}")
print(f"Thread Tile: TM={TM}, TN={TN}")
print(f"Block grid:  {num_block_m}×{num_block_n} blocks")
print(f"Warps/block: {num_warp_m}×{num_warp_n} = {num_warp_m*num_warp_n}")
print(f"Threads/warp: {num_thr_m}×{num_thr_n} = {num_thr_m*num_thr_n}")
print(f"Logical threads/block: {threads_per_block}")

# ──────────────────────────────────────────────
# 3. Print all matrices
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("INPUT MATRICES (all elements — 全局内存 GMEM / HBM)")
print("=" * 72)
print("\nMatrix A (16×16):")
for i in range(M):
    row_str = "  ".join(f"{A[i, j]:+3.0f}" for j in range(K))
    print(f"  A[{i},:] = [{row_str}]")
print("\nMatrix B (16×16):")
for i in range(N):
    row_str = "  ".join(f"{B[i, j]:+3.0f}" for j in range(N))
    print(f"  B[{i},:] = [{row_str}]")
print("\nExpected C = A @ B (16×16):")
for i in range(M):
    row_str = "  ".join(f"{C_expected[i, j]:+4.0f}" for j in range(N))
    print(f"  C[{i},:] = [{row_str}]")

# ──────────────────────────────────────────────
# 4. Full hierarchy simulation
# ──────────────────────────────────────────────

C_sim = np.zeros((M, N), dtype=np.float32)

print("\n" + "=" * 72)
print("LEVEL 0 — Thread Block Grid (global view)")
print("=" * 72)
print(f"\nC (16×16) is partitioned into {num_block_m}×{num_block_n} = {num_block_m*num_block_n} blocks of size {BM}×{BN}:")
print(f"  Block (0,0): C[0:{BM}, 0:{BN}]   — top-left    8×8")
print(f"  Block (0,1): C[0:{BM}, {BN}:{2*BN}] — top-right   8×8")
print(f"  Block (1,0): C[{BM}:{2*BM}, 0:{BN}] — bottom-left 8×8")
print(f"  Block (1,1): C[{BM}:{2*BM}, {BN}:{2*BN}] — bottom-right 8×8")

# Iterate over all blocks
for bi in range(num_block_m):
    for bj in range(num_block_n):
        m_start = bi * BM
        n_start = bj * BN

        print(f"\n{'─'*72}")
        print(f"BLOCK ({bi},{bj}) — C[{m_start}:{m_start+BM}, {n_start}:{n_start+BN}]")
        print(f"{'─'*72}")

        # Shared memory accumulator for this block (8×8)
        C_block = np.zeros((BM, BN), dtype=np.float32)

        # K dimension loop (K/BK = 4 iterations)
        for k_block in range(K // BK):
            k_start = k_block * BK

            print(f"\n  ┌─ K iteration {k_block} (k = {k_start}:{k_start+BK}) ──────────────────────┐")
            print(f"  │                     [GMEM ──load──► SMEM]                             │")

            # Load A tile (BM×BK = 8×4) from global memory to shared memory
            A_tile = A[m_start:m_start+BM, k_start:k_start+BK].copy()
            print(f"  │  A_tile (8×4) loaded: 全局内存(GMEM) ──► 共享内存(SMEM)               │")
            for i_local in range(BM):
                row_str = "  ".join(f"{A_tile[i_local, j]:+3.0f}" for j in range(BK))
                print(f"  │    row {i_local}: [{row_str}]  │")

            # Load B tile (BK×BN = 4×8) from global memory to shared memory
            B_tile = B[k_start:k_start+BK, n_start:n_start+BN].copy()
            print(f"  │  B_tile (4×8) loaded: 全局内存(GMEM) ──► 共享内存(SMEM)               │")
            for kk in range(BK):
                row_str = "  ".join(f"{B_tile[kk, j]:+3.0f}" for j in range(BN))
                print(f"  │    row {kk}: [{row_str}]  │")

            # Block-level outer product: (8×4) @ (4×8) → (8×8)
            block_contrib = A_tile @ B_tile
            C_block += block_contrib

            print(f"  │  Block contribution A_tile @ B_tile (8×8):                          │")
            for i_local in range(BM):
                row_str = "  ".join(f"{block_contrib[i_local, j]:+4.0f}" for j in range(BN))
                print(f"  │    row {i_local}: [{row_str}]  │")

            # ─── WARP LEVEL ───
            print(f"  │                                                                      │")
            print(f"  │  ── WARP DECOMPOSITION ──                                           │")
            print(f"  │           [SMEM ──load──► RF (寄存器)]                               │")
            print(f"  │  This 8×8 block is divided into {num_warp_m}×{num_warp_n} = {num_warp_m*num_warp_n} warp tiles (4×4 each):│")

            for wi in range(num_warp_m):
                for wj in range(num_warp_n):
                    wm_s = wi * WM
                    wn_s = wj * WN
                    warp_contrib = block_contrib[wm_s:wm_s+WM, wn_s:wn_s+WN]
                    print(f"  │    Warp ({wi},{wj}): C_block[{wm_s}:{wm_s+WM}, {wn_s}:{wn_s+WN}] 4×4 =")
                    for wi_local in range(WM):
                        row_str = "  ".join(f"{warp_contrib[wi_local, j]:+4.0f}" for j in range(WN))
                        print(f"  │      [{row_str}]")

            print(f"  └{'─'*70}┘")

        print(f"\n  >>> Block ({bi},{bj}) accumulated result (8×8) — 写回 全局内存(GMEM):")
        for i_local in range(BM):
            row_str = "  ".join(f"{C_block[i_local, j]:+4.0f}" for j in range(BN))
            print(f"      row {i_local}: [{row_str}]")

        C_sim[m_start:m_start+BM, n_start:n_start+BN] = C_block

# ──────────────────────────────────────────────
# 5. Thread-level detail for one representative block
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("LEVEL 2 — THREAD TILE (detailed walkthrough for Block (0,0))")
print("=" * 72)
print(f"\nEach warp tile (4×4) is decomposed into {num_thr_m}×{num_thr_n} = {num_thr_m*num_thr_n} thread tiles (2×2 each).")
print(f"Each thread computes a 2×2 tile using FMA instructions in RF (寄存器).")

# Recompute Block (0,0) with thread-level tracking
m_start, n_start = 0, 0
print(f"\nThread layout for Warp (0,0) — C[0:4, 0:4]")
print(f"  Thread (0,0) → C[0:2, 0:2]")
print(f"  Thread (0,1) → C[0:2, 2:4]")
print(f"  Thread (1,0) → C[2:4, 0:2]")
print(f"  Thread (1,1) → C[2:4, 2:4]")

# Track each thread's accumulator
thread_accum = np.zeros((num_thr_m, num_thr_n, TM, TN), dtype=np.float32)

print(f"\nNow walk through ALL 4 K-iterations for Thread (0,0) — C[0:2, 0:2]")
print(f"  Memory flow per thread iteration:")
print(f"    GMEM ──► SMEM  (Block-level load, done once per K-iteration for all threads)")
print(f"    SMEM ──► RF    (Warp/Thread loads its own A_frag and B_frag from SMEM to registers)")
print(f"    RF   ──► FMA   (Thread computes A_frag @ B_frag in registers)")

for k_block in range(K // BK):
    k_start = k_block * BK
    # A fragment for this thread: from A[0:2, k_start:k_start+BK]
    # In CUTLASS, the thread loads columns from A. Here we load TM×BK=2×4
    A_frag = A[0:TM, k_start:k_start+BK].copy()
    # B fragment: from B[k_start:k_start+BK, 0:TN]. Here BK×TN=4×2
    B_frag = B[k_start:k_start+BK, 0:TN].copy()

    step_contrib = A_frag @ B_frag
    thread_accum[0, 0] += step_contrib

    print(f"\n  K-iteration {k_block} (k=[{k_start}:{k_start+BK}]):")
    print(f"    [SMEM ──► RF] A fragment (2×4) loaded from 共享内存(SMEM) to 寄存器(RF):")
    for ti in range(TM):
        row_str = "  ".join(f"{A_frag[ti, j]:+3.0f}" for j in range(BK))
        print(f"      A_frag[{ti},:] = [{row_str}]")
    print(f"    [SMEM ──► RF] B fragment (4×2) loaded from 共享内存(SMEM) to 寄存器(RF):")
    for kk in range(BK):
        row_str = "  ".join(f"{B_frag[kk, j]:+3.0f}" for j in range(TN))
        print(f"      B_frag[{kk},:] = [{row_str}]")
    print(f"    [RF: FMA] This iteration contribution (2×4)@(4×2) → (2×2):")
    for ti in range(TM):
        row_str = "  ".join(f"{step_contrib[ti, j]:+4.0f}" for j in range(TN))
        print(f"      [{row_str}]")
    print(f"    [RF] Running accumulator for Thread (0,0):")
    for ti in range(TM):
        row_str = "  ".join(f"{thread_accum[0, 0, ti, j]:+4.0f}" for j in range(TN))
        print(f"      [{row_str}]")

# Now do the same for ALL threads
print(f"\n\nFinal thread accumulators for Warp (0,0) — C[0:4, 0:4] (RF → 写回 GMEM):")
print(f"  (This shows which 2×2 tile each of the 4 threads produces)")

full_thread_accum = {
    (0, 0): np.zeros((TM, TN), dtype=np.float32),
    (0, 1): np.zeros((TM, TN), dtype=np.float32),
    (1, 0): np.zeros((TM, TN), dtype=np.float32),
    (1, 1): np.zeros((TM, TN), dtype=np.float32),
}

for (ti, tj), acc in full_thread_accum.items():
    m_row = ti * TM
    n_col = tj * TN
    for k_block in range(K // BK):
        k_start = k_block * BK
        A_frag = A[m_row:m_row+TM, k_start:k_start+BK]
        B_frag = B[k_start:k_start+BK, n_col:n_col+TN]
        acc[:, :] += A_frag @ B_frag

print(f"  Thread (0,0) → C[0:2, 0:2] =")
for i in range(TM):
    print(f"    [{full_thread_accum[(0,0)][i,0]:+4.0f}  {full_thread_accum[(0,0)][i,1]:+4.0f}]")
print(f"  Thread (0,1) → C[0:2, 2:4] =")
for i in range(TM):
    print(f"    [{full_thread_accum[(0,1)][i,0]:+4.0f}  {full_thread_accum[(0,1)][i,1]:+4.0f}]")
print(f"  Thread (1,0) → C[2:4, 0:2] =")
for i in range(TM):
    print(f"    [{full_thread_accum[(1,0)][i,0]:+4.0f}  {full_thread_accum[(1,0)][i,1]:+4.0f}]")
print(f"  Thread (1,1) → C[2:4, 2:4] =")
for i in range(TM):
    print(f"    [{full_thread_accum[(1,1)][i,0]:+4.0f}  {full_thread_accum[(1,1)][i,1]:+4.0f}]")

# ──────────────────────────────────────────────
# 6. Simulating double-buffered shared memory (ping-pong)
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SOFTWARE PIPELINING — Double Buffering (ping-pong)")
print("=" * 72)
print("""
In real CUTLASS, while the block computes on tile k, it simultaneously
loads tile k+1 into the other shared memory buffer. This overlaps
computation and memory latency.

  Time ──────────────────────────────────────────────►
  Iter 0: [Load tile 0 → buf_A] [Compute tile 0]
  Iter 1:                     [Load tile 1 → buf_B] [Compute tile 1]
  Iter 2:                                         [Load tile 2 → buf_A] ...
  Iter 3:                                                           [Load tile 3 → buf_B] ...

This is implemented via __syncthreads() barriers and two sets of
shared memory arrays (buffer A, buffer B).
""")

# ──────────────────────────────────────────────
# 7. Epilogue — fusing bias + ReLU
# ──────────────────────────────────────────────
print("=" * 72)
print("EPILOGUE — Fusing operations after matmul (0-9 整数矩阵)")
print("=" * 72)

bias = np.array([1, 2, 3, 4, 5, 6, 7, 8,
                 9, 5, 4, 3, 2, 1, 0, -1], dtype=np.float32)

# In CUTLASS epilogue, each thread block applies bias + ReLU to its 8×8 output tile
# without writing back to global memory and re-reading.

print("\nWithout epilogue: C = A @ B")
print("With epilogue:    C = ReLU(A @ B + bias)")

C_with_epilogue = np.maximum(C_expected + bias.reshape(M, 1), 0.0)

print(f"\nBias vector (broadcast to columns):")
print(bias.reshape(M, 1))

print(f"\nBlock (0,0) — C + bias (before ReLU):")
block00_with_bias = C_expected[0:BM, 0:BN] + bias[0:BM].reshape(BM, 1)
for i in range(BM):
    row_str = "  ".join(f"{block00_with_bias[i, j]:+4.0f}" for j in range(BN))
    print(f"  row {i}: [{row_str}]")

print(f"\nBlock (0,0) — ReLU(C + bias):")
block00_final = np.maximum(block00_with_bias, 0.0)
for i in range(BM):
    row_str = "  ".join(f"{block00_final[i, j]:+4.0f}" for j in range(BN))
    print(f"  row {i}: [{row_str}]")

print("""
The key insight: each thread block computes its 8×8 output tile,
applies bias+ReLU directly in registers/shared memory, and writes
the final result once to global memory. No intermediate round-trip.

In CUTLASS, the Epilogue is a template parameter. Users can chain:
  LinearCombination → BiasAdd → ReLU → Clamp → ...
""")

# ──────────────────────────────────────────────
# 8. WMMA / Tensor Core mention
# ──────────────────────────────────────────────
print("=" * 72)
print("WMMA / TENSOR CORE ABSTRACTION")
print("=" * 72)
print("""
On NVIDIA Volta/Turing/Ampere/Hopper GPUs, the "thread tile" level is
replaced by a warp-level matrix multiply-accumulate (WMMA) or direct
mma.sync PTX instruction, which computes a small matrix multiply in
one operation using Tensor Cores.

  Thread Tile (scalar FMA)  →  WMMA / mma.sync (Tensor Core)

The same hierarchical structure applies at the block and warp levels;
only the innermost compute primitive changes.
""")

# ──────────────────────────────────────────────
# 9. Final verification
# ──────────────────────────────────────────────
print("=" * 72)
print("VERIFICATION")
print("=" * 72)
print("\nSimulated C (from hierarchy):")
for i in range(M):
    row_str = "  ".join(f"{C_sim[i, j]:+4.0f}" for j in range(N))
    print(f"  C_sim[{i},:] = [{row_str}]")
print(f"\nMax absolute error: {np.max(np.abs(C_sim - C_expected)):.10f}")

if np.allclose(C_sim, C_expected, atol=1e-5):
    print("\n✓✓✓ PASS — Simulated result matches numpy A@B ✓✓✓")
else:
    print("\n✗✗✗ FAIL — Mismatch! ✗✗✗")
    diff = C_sim - C_expected
    print("Difference:")
    print(diff)

print(f"\nAll threads cumulative: {num_block_m * num_block_n * num_warp_m * num_warp_n * num_thr_m * num_thr_n} logical threads")
print(f"  Block grid: {num_block_m}×{num_block_n}")
print(f"  Warps per block: {num_warp_m}×{num_warp_n}")
print(f"  Threads per warp: {num_thr_m}×{num_thr_n}")

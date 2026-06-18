#!/usr/bin/env python3
"""
CUDA SGEMM Optimization Verification Script
Based on: https://siboehm.com/articles/22/CUDA-MMM
Demo: M=N=K=16, seed=42, float32

This script simulates each kernel stage and verifies against numpy ground truth.
"""

import numpy as np

np.random.seed(42)
np.set_printoptions(linewidth=220, precision=6, suppress=True, threshold=10000)

# ============================================================
# 1. Generate demo matrices
# ============================================================
M, N, K = 16, 16, 16
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
C_ref = A @ B  # Ground truth

print("=" * 80)
print("CUDA SGEMM OPTIMIZATION VERIFICATION")
print(f"Problem size: M={M}, N={N}, K={K}")
print(f"Data type: float32")
print(f"Random seed: 42")
print("=" * 80)

print("\n" + "=" * 80)
print("MATRIX A (16x16)")
print("=" * 80)
for i in range(M):
    row_str = "  ".join(f"{A[i, j]:10.6f}" for j in range(K))
    print(f"Row {i:2d}: [{row_str}]")

print("\n" + "=" * 80)
print("MATRIX B (16x16)")
print("=" * 80)
for i in range(K):
    row_str = "  ".join(f"{B[i, j]:10.6f}" for j in range(N))
    print(f"Row {i:2d}: [{row_str}]")

print("\n" + "=" * 80)
print("GROUND TRUTH C = A @ B (16x16)")
print("=" * 80)
for i in range(M):
    row_str = "  ".join(f"{C_ref[i, j]:10.6f}" for j in range(N))
    print(f"Row {i:2d}: [{row_str}]")


# ============================================================
# Helper: check correctness
# ============================================================
def check(label, computed, ref, tol=1e-4):
    diff = np.abs(computed - ref).max()
    ok = diff < tol
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {label}: max error = {diff:.6f}, tolerance = {tol}")
    if not ok:
        print(f"  Computed:\n{computed}")
        print(f"  Reference:\n{ref}")
    return ok


# ============================================================
# Kernel 1: Naive Implementation
# ============================================================
print("\n\n" + "=" * 80)
print("KERNEL 1: NAIVE IMPLEMENTATION")
print("  blockDim = (4, 4), grid = (4, 4)")
print("  Each thread computes one C[i,j] = dot(A[i,:], B[:,j])")
print("=" * 80)

def kernel1_naive(A, B, block_size=4):
    C = np.zeros((M, N), dtype=np.float32)
    grid_rows = (M + block_size - 1) // block_size
    grid_cols = (N + block_size - 1) // block_size
    for bx in range(grid_rows):
        for by in range(grid_cols):
            for tx in range(block_size):
                for ty in range(block_size):
                    i = bx * block_size + tx
                    j = by * block_size + ty
                    if i < M and j < N:
                        acc = 0.0
                        for k in range(K):
                            acc += A[i, k] * B[k, j]
                        C[i, j] = acc
    return C

C_k1 = kernel1_naive(A, B)
check("Kernel 1 (Naive)", C_k1, C_ref)

# Show sample dot products
print("\nSample dot product calculations (Kernel 1):")
for i, j in [(0, 0), (0, 1), (1, 1), (15, 15)]:
    terms = [A[i, k] * B[k, j] for k in range(K)]
    computed = sum(terms)
    print(f"\n  C[{i},{j}] = sum(A[{i},k] * B[k,{j}]) for k=0..15:")
    for k in range(K):
        print(f"    k={k:2d}: {A[i, k]:10.6f} * {B[k, j]:10.6f} = {terms[k]:12.6f}")
    print(f"    Sum = {computed:12.6f}")
    print(f"    Reference = {C_ref[i, j]:12.6f}")


# ============================================================
# Kernel 2: Global Memory Coalescing
# ============================================================
print("\n\n" + "=" * 80)
print("KERNEL 2: GLOBAL MEMORY COALESCING")
print("  blockDim = (16), grid = (16, 16)")
print("  1D thread indexing to enable coalesced memory access")
print("=" * 80)

def kernel2_coalescing(A, B, block_size=16):
    C = np.zeros((M, N), dtype=np.float32)
    grid_rows = (M + block_size - 1) // block_size
    grid_cols = (N + block_size - 1) // block_size
    for bx in range(grid_rows):
        for by in range(grid_cols):
            for t in range(block_size * block_size):
                # 1D thread index -> 2D position for coalescing
                i = bx * block_size + (t // block_size)
                j = by * block_size + (t % block_size)
                if i < M and j < N:
                    acc = 0.0
                    for k in range(K):
                        acc += A[i, k] * B[k, j]
                    C[i, j] = acc
    return C

C_k2 = kernel2_coalescing(A, B)
check("Kernel 2 (GMEM Coalescing)", C_k2, C_ref)


# ============================================================
# Kernel 3: Shared Memory Cache-Blocking
# ============================================================
print("\n\n" + "=" * 80)
print("KERNEL 3: SHARED MEMORY CACHE-BLOCKING")
print("  BM=BN=BK=4, blockDim=(4,4)")
print("  Load tiles of A and B into shared memory, then compute")
print("=" * 80)

def kernel3_smem_caching(A, B, BM=4, BN=4, BK=4):
    C = np.zeros((M, N), dtype=np.float32)
    grid_rows = M // BM
    grid_cols = N // BN
    for bx in range(grid_rows):
        for by in range(grid_cols):
            # Shared memory allocations
            As = np.zeros((BM, BK), dtype=np.float32)
            Bs = np.zeros((BK, BN), dtype=np.float32)

            c_row_start = bx * BM
            c_col_start = by * BN

            for tx in range(BM):
                for ty in range(BN):
                    acc = 0.0
                    # Outer loop over K tiles
                    for bk_start in range(0, K, BK):
                        # Load tile into shared memory
                        for r in range(BM):
                            for c in range(BK):
                                a_row = c_row_start + r
                                a_col = bk_start + c
                                As[r, c] = A[a_row, a_col] if a_row < M and a_col < K else 0.0
                        for r in range(BK):
                            for c in range(BN):
                                b_row = bk_start + r
                                b_col = c_col_start + c
                                Bs[r, c] = B[b_row, b_col] if b_row < K and b_col < N else 0.0

                        # Compute partial dot product
                        for dot_idx in range(BK):
                            acc += As[tx, dot_idx] * Bs[dot_idx, ty]

                    C[c_row_start + tx, c_col_start + ty] = acc
    return C

C_k3 = kernel3_smem_caching(A, B)
check("Kernel 3 (SMEM Caching)", C_k3, C_ref)

# Show tile loading for a specific block
print("\nTile loading example (Bx=0, By=0, first K-tile bk_start=0):")
print("  As (4x4 shared memory tile of A, rows 0-3, cols 0-3):")
for r in range(4):
    row_str = "  ".join(f"{A[r, c]:10.6f}" for c in range(4))
    print(f"    [{row_str}]")
print("  Bs (4x4 shared memory tile of B, rows 0-3, cols 0-3):")
for r in range(4):
    row_str = "  ".join(f"{B[r, c]:10.6f}" for c in range(4))
    print(f"    [{row_str}]")

print("\n  Thread (0,0) partial dot product (bk_start=0):")
partial = sum(A[0, k] * B[k, 0] for k in range(4))
print(f"    acc += As[0,0]*Bs[0,0] + As[0,1]*Bs[1,0] + As[0,2]*Bs[2,0] + As[0,3]*Bs[3,0]")
for k in range(4):
    print(f"    k={k}: {A[0, k]:10.6f} * {B[k, 0]:10.6f} = {A[0,k]*B[k,0]:12.6f}")
print(f"    Partial sum (bk=0): {partial:12.6f}")
print(f"    Full C[0,0] = {C_ref[0,0]:12.6f} (need 3 more K-tiles)")


# ============================================================
# Kernel 4: 1D Blocktiling
# ============================================================
print("\n\n" + "=" * 80)
print("KERNEL 4: 1D BLOCKTILING")
print("  BM=8, BN=8, BK=4, TM=4")
print("  Each thread computes TM=4 results (1D strip in C)")
print("=" * 80)

def kernel4_1d_blocktiling(A, B, BM=8, BN=8, BK=4, TM=4):
    C = np.zeros((M, N), dtype=np.float32)
    grid_rows = M // BM
    grid_cols = N // BN
    threads_per_block = (BM // TM) * BN  # Each thread handles TM rows, 1 col
    for bx in range(grid_rows):
        for by in range(grid_cols):
            c_row_start = bx * BM
            c_col_start = by * BN

            # Shared memory
            As = np.zeros((BM, BK), dtype=np.float32)
            Bs = np.zeros((BK, BN), dtype=np.float32)

            # Each thread: threadRow handles TM results in C
            num_thread_rows = BM // TM
            for thr in range(threads_per_block):
                threadRow = thr // BN
                threadCol = thr % BN
                # threadRow indexes groups of TM rows
                res_row_start = threadRow * TM

                threadResults = np.zeros(TM, dtype=np.float32)

                for bk_start in range(0, K, BK):
                    # Load As
                    for r in range(BM):
                        for c in range(BK):
                            a_row = c_row_start + r
                            a_col = bk_start + c
                            As[r, c] = A[a_row, a_col] if a_row < M and a_col < K else 0.0
                    # Load Bs
                    for r in range(BK):
                        for c in range(BN):
                            b_row = bk_start + r
                            b_col = c_col_start + c
                            Bs[r, c] = B[b_row, b_col] if b_row < K and b_col < N else 0.0

                    for dotIdx in range(BK):
                        Btmp = Bs[dotIdx, threadCol]
                        for resIdx in range(TM):
                            a_row = res_row_start + resIdx
                            threadResults[resIdx] += As[a_row, dotIdx] * Btmp

                for resIdx in range(TM):
                    C[c_row_start + res_row_start + resIdx, c_col_start + threadCol] = threadResults[resIdx]
    return C

C_k4 = kernel4_1d_blocktiling(A, B)
check("Kernel 4 (1D Blocktiling)", C_k4, C_ref)

# Show thread computation
print("\nSample: Block (0,0), Thread (threadRow=0, threadCol=0)")
print("  This thread computes C[0,0], C[1,0], C[2,0], C[3,0]")
print("  TM=4 results, BK=4 dot products per K-tile, 4 K-tiles total")
print(f"  Expected: C[0,0]={C_ref[0,0]:.6f}, C[1,0]={C_ref[1,0]:.6f}, C[2,0]={C_ref[2,0]:.6f}, C[3,0]={C_ref[3,0]:.6f}")


# ============================================================
# Kernel 5: 2D Blocktiling
# ============================================================
print("\n\n" + "=" * 80)
print("KERNEL 5: 2D BLOCKTILING")
print("  BM=8, BN=8, BK=4, TM=4, TN=4")
print("  Each thread computes TM*TN=16 results (4x4 tile in C)")
print("=" * 80)

def kernel5_2d_blocktiling(A, B, BM=8, BN=8, BK=4, TM=4, TN=4):
    C = np.zeros((M, N), dtype=np.float32)
    grid_rows = M // BM
    grid_cols = N // BN
    num_thread_rows = BM // TM
    num_thread_cols = BN // TN
    for bx in range(grid_rows):
        for by in range(grid_cols):
            c_row_start = bx * BM
            c_col_start = by * BN

            As = np.zeros((BM, BK), dtype=np.float32)
            Bs = np.zeros((BK, BN), dtype=np.float32)

            for thr_row in range(num_thread_rows):
                for thr_col in range(num_thread_cols):
                    res_row_start = thr_row * TM
                    res_col_start = thr_col * TN

                    threadResults = np.zeros((TM, TN), dtype=np.float32)

                    for bk_start in range(0, K, BK):
                        for r in range(BM):
                            for c in range(BK):
                                a_row = c_row_start + r
                                a_col = bk_start + c
                                As[r, c] = A[a_row, a_col] if a_row < M and a_col < K else 0.0
                        for r in range(BK):
                            for c in range(BN):
                                b_row = bk_start + r
                                b_col = c_col_start + c
                                Bs[r, c] = B[b_row, b_col] if b_row < K and b_col < N else 0.0

                        for dotIdx in range(BK):
                            for resRow in range(TM):
                                for resCol in range(TN):
                                    a_row = res_row_start + resRow
                                    b_col = res_col_start + resCol
                                    threadResults[resRow, resCol] += As[a_row, dotIdx] * Bs[dotIdx, b_col]

                    for resRow in range(TM):
                        for resCol in range(TN):
                            C[c_row_start + res_row_start + resRow,
                              c_col_start + res_col_start + resCol] = threadResults[resRow, resCol]
    return C

C_k5 = kernel5_2d_blocktiling(A, B)
check("Kernel 5 (2D Blocktiling)", C_k5, C_ref)

print("\nSample: Block (0,0), Thread (thr_row=0, thr_col=0)")
print("  This thread computes C[0:4, 0:4] = 4x4 sub-tile")
print(f"  Expected 4x4 block:")
for i in range(4):
    row_str = "  ".join(f"{C_ref[i, j]:10.6f}" for j in range(4))
    print(f"    [{row_str}]")


# ============================================================
# Kernel 6: Vectorized Memory Access (float4)
# ============================================================
print("\n\n" + "=" * 80)
print("KERNEL 6: VECTORIZED MEMORY ACCESS (float4)")
print("  Same as K5 but loads 4 floats at once (float4)")
print("  Simulated by loading 4 consecutive elements per access")
print("=" * 80)

def kernel6_vectorized(A, B, BM=8, BN=8, BK=4, TM=4, TN=4):
    C = np.zeros((M, N), dtype=np.float32)
    grid_rows = M // BM
    grid_cols = N // BN
    num_thread_rows = BM // TM
    num_thread_cols = BN // TN
    for bx in range(grid_rows):
        for by in range(grid_cols):
            c_row_start = bx * BM
            c_col_start = by * BN

            As = np.zeros((BM, BK), dtype=np.float32)
            Bs = np.zeros((BK, BN), dtype=np.float32)

            for thr_row in range(num_thread_rows):
                for thr_col in range(num_thread_cols):
                    res_row_start = thr_row * TM
                    res_col_start = thr_col * TN
                    threadResults = np.zeros((TM, TN), dtype=np.float32)

                    for bk_start in range(0, K, BK):
                        # Vectorized load of As (load 4 consecutive column elements as float4)
                        # As has shape (BM, BK) where BK=4, so each row is a float4
                        for r in range(BM):
                            for c in range(0, BK, 4):
                                for cc in range(4):
                                    a_row = c_row_start + r
                                    a_col = bk_start + c + cc
                                    As[r, c + cc] = A[a_row, a_col] if a_row < M and a_col < K else 0.0
                        # Vectorized load of Bs (load 4 consecutive column elements as float4)
                        for r in range(BK):
                            for c in range(0, BN, 4):
                                for cc in range(4):
                                    b_row = bk_start + r
                                    b_col = c_col_start + c + cc
                                    Bs[r, c + cc] = B[b_row, b_col] if b_row < K and b_col < N else 0.0

                        for dotIdx in range(BK):
                            for resRow in range(TM):
                                for resCol in range(TN):
                                    a_row = res_row_start + resRow
                                    b_col = res_col_start + resCol
                                    threadResults[resRow, resCol] += As[a_row, dotIdx] * Bs[dotIdx, b_col]

                    for resRow in range(TM):
                        for resCol in range(TN):
                            C[c_row_start + res_row_start + resRow,
                              c_col_start + res_col_start + resCol] = threadResults[resRow, resCol]
    return C

C_k6 = kernel6_vectorized(A, B)
check("Kernel 6 (Vectorized)", C_k6, C_ref)


# ============================================================
# Kernel 10: Warptiling
# ============================================================
print("\n\n" + "=" * 80)
print("KERNEL 10: WARPTILING")
print("  WM=WN=4, WMIter=WNIter=2, TM=TN=4")
print("  Warps divide work: 2x2 warp grid, each warp handles 4x4 sub-tile")
print("=" * 80)

def kernel10_warptiling(A, B, BM=8, BN=8, BK=8, WM=4, WN=4, WMIter=2, WNIter=2, TM=4, TN=4):
    """Warptiling kernel.
    
    Warp organizes work: each warp handles a region of C.
    Warp tile size = WM * WN = 4x4 elements (cooperatively loaded per step)
    Warp iterates WMIter x WNIter times, each handling TM x TN elements per thread.
    Total warp work = WMIter*TM x WNIter*TN = 8x8 elements per warp.
    
    Since BM=8, BN=8, and each warp covers 8x8, we only need 1 warp per block.
    Effectively the whole block is handled by one warp in 4 "thread-like" iterations.
    """
    C = np.zeros((M, N), dtype=np.float32)
    grid_rows = M // BM
    grid_cols = N // BN
    
    # Effective warp region: each warp covers WMIter*TM x WNIter*TN
    warp_region_m = WMIter * TM  # 8
    warp_region_n = WNIter * TN  # 8
    warps_m = max(1, BM // warp_region_m)
    warps_n = max(1, BN // warp_region_n)
    
    for bx in range(grid_rows):
        for by in range(grid_cols):
            c_row_start = bx * BM
            c_col_start = by * BN

            As = np.zeros((BM, BK), dtype=np.float32)
            Bs = np.zeros((BK, BN), dtype=np.float32)

            for warp_i in range(warps_m):
                for warp_j in range(warps_n):
                    warp_row_start = warp_i * warp_region_m
                    warp_col_start = warp_j * warp_region_n

                    # Within each warp, WMIter x WNIter threads compute TM x TN tiles
                    for t_row in range(WMIter):
                        for t_col in range(WNIter):
                            res_row_start = warp_row_start + t_row * TM
                            res_col_start = warp_col_start + t_col * TN

                            threadResults = np.zeros((TM, TN), dtype=np.float32)

                            for bk_start in range(0, K, BK):
                                # Cooperative load of As tile into shared memory (simulating warp-cooperative load)
                                for r in range(BM):
                                    for c in range(BK):
                                        a_row = c_row_start + r
                                        a_col = bk_start + c
                                        As[r, c] = A[a_row, a_col] if a_row < M and a_col < K else 0.0
                                # Cooperative load of Bs tile
                                for r in range(BK):
                                    for c in range(BN):
                                        b_row = bk_start + r
                                        b_col = c_col_start + c
                                        Bs[r, c] = B[b_row, b_col] if b_row < K and b_col < N else 0.0

                                # Compute: thread's TM x TN sub-tile
                                for dotIdx in range(BK):
                                    for resRow in range(TM):
                                        for resCol in range(TN):
                                            a_val = As[res_row_start + resRow, dotIdx]
                                            b_val = Bs[dotIdx, res_col_start + resCol]
                                            threadResults[resRow, resCol] += a_val * b_val

                            for resRow in range(TM):
                                for resCol in range(TN):
                                    rr = c_row_start + res_row_start + resRow
                                    cc = c_col_start + res_col_start + resCol
                                    C[rr, cc] = threadResults[resRow, resCol]
    return C

C_k10 = kernel10_warptiling(A, B)
check("Kernel 10 (Warptiling)", C_k10, C_ref)

print("\nWarp work division (Block 0,0):")
print("  BM=8, BN=8, WMIter=WNIter=2, TM=TN=4")
print("  Warp region = WMIter*TM x WNIter*TN = 8x8")
print("  1 warp per block (covers entire 8x8 block):")
print("    Warp (0,0) -> C[0:8, 0:8]")
print("  Within warp: WMIter x WNIter = 4 threads, each computing TM x TN = 4x4 tile:")
print("    Thread (0,0) -> C[0:4, 0:4]")
print("    Thread (0,1) -> C[0:4, 4:8]")
print("    Thread (1,0) -> C[4:8, 0:4]")
print("    Thread (1,1) -> C[4:8, 4:8]")


# ============================================================
# Arithmetic Intensity & Memory Calculations
# ============================================================
print("\n\n" + "=" * 80)
print("PERFORMANCE ANALYSIS")
print("=" * 80)

total_flops = 2 * M * N * K  # multiply + add
total_gmem_reads_min = (M * K + K * N) * 4  # bytes (fp32)
total_gmem_writes = M * N * 4
total_gmem_min = total_gmem_reads_min + total_gmem_writes

print(f"\nTotal FLOPs: {total_flops} (2*M*N*K)")
print(f"Minimum GMEM reads: {total_gmem_reads_min} bytes ({M}*{K}*4 + {K}*{N}*4)")
print(f"GMEM writes: {total_gmem_writes} bytes ({M}*{N}*4)")
print(f"Minimum total GMEM traffic: {total_gmem_min} bytes")
print(f"Arithmetic Intensity (min): {total_flops / total_gmem_min:.3f} FLOPs/Byte")

# Per-kernel analysis
print("\n--- Per-kernel analysis for our 16x16 demo ---")

# K1 Naive: each thread loads entire row of A and column of B
k1_reads = M * N * 2 * K * 4  # Each of 256 threads loads 2*16 floats
print(f"\nK1 Naive: ~{k1_reads} bytes GMEM reads (worst case, no cache)")

# K2: Same FLOPs, but coalesced access
print(f"K2 Coalescing: Same FLOPs, coalesced reads ~{M*N*2*K*4} bytes")

# K3: SMEM caching
# Each block loads BM*BK + BK*BN floats per K-tile
# BM=BN=BK=4, blocks=4*4=16, K-tiles=16/4=4
k3_loads_per_block_per_tile = (4 * 4 + 4 * 4) * 4  # bytes
k3_total_smem_loads = 16 * 4 * k3_loads_per_block_per_tile
print(f"K3 SMEM Caching: ~{k3_total_smem_loads} bytes GMEM reads")

# K4: 1D Blocktiling
# BM=8, BN=8, BK=4. Blocks = 2*2=4. K-tiles per block = 16/4=4
k4_loads_per_block_per_tile = (8 * 4 + 4 * 8) * 4
k4_total_smem_loads = 4 * 4 * k4_loads_per_block_per_tile
print(f"K4 1D Blocktiling: ~{k4_total_smem_loads} bytes GMEM reads")

# K5: 2D Blocktiling
# BM=8, BN=8, BK=4. Same memory as K4 but more work per thread
k5_loads_per_block_per_tile = (8 * 4 + 4 * 8) * 4
k5_total_smem_loads = 4 * 4 * k5_loads_per_block_per_tile
k5_ai = total_flops / k5_total_smem_loads
print(f"K5 2D Blocktiling: ~{k5_total_smem_loads} bytes GMEM reads")
print(f"  Arithmetic Intensity: {k5_ai:.3f} FLOPs/Byte")

# Occupancy example calculation
print("\n--- Occupancy Example (A6000-like) ---")
print("  SMEM per block: 8KB (for BM=BN=BK=32 in original article)")
print("  Registers per thread: 37")
print("  Threads per block: 1024")
print("  Max threads per SM: 1536")
print("  Max SMEM per SM: 100KB")
print("  Occupancy = 1024/1536 = 66.7% (limited by threads per SM)")

# Roofline model summary
print("\n--- Roofline Model Summary ---")
print("  Peak fp32 compute: ~30 TFLOPs/s (A6000)")
print("  Peak GMEM bandwidth: ~768 GB/s (A6000)")
print("  Ridge point: 30e12 / 768e9 = 39 FLOPs/Byte")
print("  K3 AI: ~0.5 FLOPs/Byte -> memory bound")
print("  K5 AI: ~2.0 FLOPs/Byte -> still memory bound but improving")
print("  K10 AI: ~8.0 FLOPs/Byte -> approaching compute-bound region")


# ============================================================
# Final Summary
# ============================================================
print("\n\n" + "=" * 80)
print("FINAL VERIFICATION SUMMARY")
print("=" * 80)

results = [
    ("Kernel 1 (Naive)", C_k1, C_ref),
    ("Kernel 2 (GMEM Coalescing)", C_k2, C_ref),
    ("Kernel 3 (SMEM Caching)", C_k3, C_ref),
    ("Kernel 4 (1D Blocktiling)", C_k4, C_ref),
    ("Kernel 5 (2D Blocktiling)", C_k5, C_ref),
    ("Kernel 6 (Vectorized)", C_k6, C_ref),
    ("Kernel 10 (Warptiling)", C_k10, C_ref),
]

all_pass = True
for name, computed, ref in results:
    diff = np.abs(computed - ref).max()
    ok = diff < 1e-4
    all_pass = all_pass and ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max error = {diff:.6f}")

print(f"\n{'ALL CHECKS PASSED!' if all_pass else 'SOME CHECKS FAILED!'}")
print("=" * 80)

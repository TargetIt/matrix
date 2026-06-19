#!/usr/bin/env python3
"""
CUDA SGEMM 优化验证脚本
基于 wangzyon/NVIDIA_SGEMM_PRACTICE
整数矩阵 (seed=1, values -5..5), dtype=float32
演示问题规模: M=N=K=16

每个 kernel 阶段:
  K1 Naive: blockDim=(4,4), gridDim=(4,4,1)
  K2 SMEM Caching: BM=BN=BK=4, blockDim=(4,4), gridDim=(4,4)
  K3 1D Thread Tile: BM=8,BN=8,BK=4,TM=4, blockDim=(8,2), gridDim=(2,2)
  K4 2D Thread Tile: BM=8,BN=8,BK=4,TM=4,TN=4, blockDim=(2,2), gridDim=(2,2)
  K5 Register Caching: 同 K4 + 寄存器缓存
  K6 Float4 Vectorized: 同 K5 + float4 向量化加载
  K7 Double Buffering: 同 K6 + 双缓冲
"""

import numpy as np

np.random.seed(1)

A = np.random.randint(-5, 6, (16, 16)).astype(np.float32)
B = np.random.randint(-5, 6, (16, 16)).astype(np.float32)
C_ref = A @ B

print("=" * 70)
print("CUDA SGEMM 优化验证")
print("整数矩阵 (seed=1, values -5..5), dtype=float32")
print(f"M=N=K=16, 总 FLOPs = 16*16*16*2 = {16*16*16*2}")
print("=" * 70)

# ---- 矩阵 A ----
print("\n>>> 矩阵 A (16x16):")
print("     ", end="")
for j in range(16):
    print(f"{'col':>3s}{j:<3d}", end="")
print()
for i in range(16):
    print(f"row{i:2d} ", end="")
    for j in range(16):
        print(f"{int(A[i,j]):5d} ", end="")
    print()

# ---- 矩阵 B ----
print("\n>>> 矩阵 B (16x16):")
print("     ", end="")
for j in range(16):
    print(f"{'col':>3s}{j:<3d}", end="")
print()
for i in range(16):
    print(f"row{i:2d} ", end="")
    for j in range(16):
        print(f"{int(B[i,j]):5d} ", end="")
    print()

# ---- 参考结果 C ----
print("\n>>> 参考结果 C = A@B (16x16):")
print("     ", end="")
for j in range(16):
    print(f"{'col':>3s}{j:<3d}", end="")
print()
for i in range(16):
    print(f"row{i:2d} ", end="")
    for j in range(16):
        print(f"{int(C_ref[i,j]):5d} ", end="")
    print()

# verify exact matrices
A_expected = np.array([
    [ 0,  3,  4,  0, -5, -5, -4,  2,  1,  4, -3, -1,  0, -3, -1,  5],
    [-3, -1,  2,  2,  4, -4,  2, -5,  1,  4,  4,  2,  1,  4, -4, -5],
    [-4,  3,  3, -2,  5,  4,  3,  2, -2,  1,  0, -4,  4, -2, -1,  3],
    [-4,  5, -1, -5, -2,  4, -3, -5, -1,  4, -3,  2,  2,  5,  4,  3],
    [ 1,  4, -2,  2,  2, -1,  0,  4, -2,  1,  3, -5, -3,  5,  2,  2],
    [ 4,  2, -2,  5, -5,  3,  2,  2, -4, -4, -2, -5,  3,  1, -1,  0],
    [ 1, -3,  0,  2,  3, -1, -1,  2,  2, -1,  4, -5, -3, -5,  5,  2],
    [-4,  2,  4,  3,  5, -1, -5, -4,  4,  3, -3, -2, -4, -3,  2, -3],
    [ 5,  1, -5,  4, -3,  1,  1, -3,  2,  2, -5,  1,  0, -4,  5, -1],
    [ 1, -5,  1,  0, -4, -3, -4,  0, -1,  5, -5,  2,  3,  4,  0,  2],
    [-5,  4, -2,  4, -4, -1, -1,  1,  3,  3,  4, -3,  2,  0,  0, -1],
    [ 0,  3,  0,  3, -4, -4,  3,  2, -5,  5, -2, -1, -3, -5, -2,  0],
    [-4, -3, -1, -2, -5,  1, -5,  2, -3,  5,  3, -2, -5,  3, -1, -3],
    [ 4, -5, -2,  3, -4,  5, -1, -2, -2,  5,  1,  2, -2,  0, -2, -3],
    [-1, -1, -5, -2, -2,  3, -2,  0,  1,  2,  0, -4,  2, -5,  5,  5],
    [-3,  3, -3, -4, -1, -5, -1, -4,  2, -2, -4,  1,  1,  4,  5,  1],
], dtype=np.float32)

B_expected = np.array([
    [ 4,  1, -5, -5, -3,  4,  1, -5,  1,  2, -5, -2,  4, -5, -2,  5],
    [-1,  2,  0, -2,  3,  3, -5,  1,  2,  4,  0, -1,  4,  5,  0, -3],
    [ 0,  1,  1,  3,  5,  2,  2,  2, -3,  1, -5,  0, -3, -4,  5,  3],
    [ 0,  4, -1,  4, -4, -3, -5,  5, -1,  2, -5,  1, -3, -1, -2,  1],
    [ 2,  1, -2, -5,  1, -1,  2,  1, -3,  5,  4,  5,  0,  4,  4,  4],
    [ 3,  1, -1, -3,  5,  4, -1, -5, -5, -2,  5, -1,  4, -2,  4, -4],
    [-3,  0, -1, -5,  3, -3, -2,  4,  4, -1, -1,  3, -3, -4,  1, -2],
    [ 3,  4,  2, -5,  0, -3, -3,  5,  5,  5,  3,  0,  5, -5,  0,  4],
    [ 3,  5,  1,  1, -5, -1,  2, -2, -5, -4,  1,  5, -5,  1, -4,  1],
    [-1, -3,  0, -1,  1, -3,  4, -3,  2,  0, -5,  2,  5,  3,  3,  3],
    [-5,  2, -3, -5,  2, -4,  5, -4,  5,  4,  0, -4,  0,  4,  5,  1],
    [-1,  4,  3,  2,  5,  0, -4,  5,  3, -5,  0, -2,  4, -5, -1,  3],
    [ 1, -3, -1, -2,  5, -3, -5, -5, -1,  5,  5, -3,  0, -5, -5, -2],
    [ 3,  0, -2, -4, -1,  2, -2, -3, -3,  5, -3,  1,  1,  5, -5,  5],
    [-4,  0,  5,  1,  0,  3,  3,  0,  0,  2,  0,  4, -4, -2,  4, -2],
    [-2, -2,  1, -4, -2, -5,  0, -5,  0, -3,  2,  1, -1,  5, -5, -3],
], dtype=np.float32)

C_expected = np.array([
    [-10, -19,  40,  57, -32, -27,   0,   2,  22, -28, -55, -10,   9,  13, -44,  -2],
    [-20,   1, -40,  17,  26, -39,  24,  23,  -6,  31, -62,  12, -27,  58,  24,  68],
    [ -7, -33,  -4, -71,  85, -34, -12, -22, -16,  54,  83,  18,   8,  23,  52, -52],
    [-18, -60,  42,   7,  63,  56, -10, -89, -50, -23,  30,   4,  41,  86, -17, -63],
    [ -4,  10, -25, -86, -44, -10,   6, -14,  32, 105, -23,  21,  26,  95,   7,  27],
    [ 30,  -9, -41, -25, -19,  22, -92, -14,   8,  52,  -8, -49,  16, -58, -59, -47],
    [-31,  19,   8, -12, -67, -32,  89,  -6,   6,  19,  10,  44, -67,  16,  65,  -5],
    [  3,  19,  33, 109, -33,  33,  57,  44, -87, -18, -35,  76, -59,  70,  71,   9],
    [  2,   4,  16,  48, -55,  38, -24,  -1,  -5, -56, -35,  27,  -6, -59, -33, -40],
    [ 30, -50,  22,  50, -27, -10,  -7, -33, -20, -24, -56,  -8,  21, -29, -82,  50],
    [-36,  14,  10,  24, -18, -53, -17,  -4,  23,  32,  -6,  -8, -17,  62, -14, -39],
    [-45, -21,  11,  35,  -7, -36, -23,  84,  97, -14, -78, -16,  37, -17,  23, -14],
    [-12, -26,  12,  34, -15,   6,  65, -15,  25,  -6, -37, -34,  52,  60,  51,  23],
    [ 19,  -6, -35,  22,  -8,  19,  22, -37,   3, -50, -64, -41,  60, -46,  22,  27],
    [-30, -48,  32, -13, -28, -24,  42, -89,  -3, -33,  75,  13, -10,  23,  -1, -94],
    [-23, -31,  50,  38, -26,  34, -13,  -7, -24, -16,  17,  38, -49,  61, -71, -35],
], dtype=np.float32)

print(f"\n>>> 矩阵验证 (A): {'PASS' if np.array_equal(A, A_expected) else 'FAIL'}")
print(f">>> 矩阵验证 (B): {'PASS' if np.array_equal(B, B_expected) else 'FAIL'}")
print(f">>> 矩阵验证 (C_ref): {'PASS' if np.array_equal(C_ref, C_expected) else 'FAIL'}")

print(f"\nC[0,0]={int(C_ref[0,0]):d}  C[0,15]={int(C_ref[0,15]):d}")
print(f"C[15,0]={int(C_ref[15,0]):d}  C[15,15]={int(C_ref[15,15]):d}")

total_flops = 16 * 16 * 16 * 2

# ============================================================
# 1. K1: Naive
# ============================================================
print("\n" + "=" * 70)
print("K1: Naive Kernel (纯全局内存)")
print("blockDim=(4,4), gridDim=(4,4,1)")
print("=" * 70)

C_k1 = np.zeros((16, 16), dtype=np.float32)
gA_k1, gB_k1, gC_k1 = 0, 0, 0

for block_row in range(4):
    for block_col in range(4):
        for thread_row in range(4):
            for thread_col in range(4):
                row = block_row * 4 + thread_row
                col = block_col * 4 + thread_col
                acc = np.float32(0.0)
                for k in range(16):
                    acc += A[row, k] * B[k, col]
                    gA_k1 += 1  # GMEM read
                    gB_k1 += 1  # GMEM read
                C_k1[row, col] = acc
                gC_k1 += 1  # GMEM write

gmem_bytes_k1 = (gA_k1 + gB_k1 + gC_k1) * 4
smem_bytes_k1 = 0
ai_k1_gmem = total_flops / gmem_bytes_k1

print(f"\nK1 全局内存 (GMEM) 读取 A: {gA_k1} floats = {gA_k1*4} bytes")
print(f"K1 全局内存 (GMEM) 读取 B: {gB_k1} floats = {gB_k1*4} bytes")
print(f"K1 全局内存 (GMEM) 写入 C: {gC_k1} floats = {gC_k1*4} bytes")
print(f"K1 共享内存 (SMEM) 访问: 0 bytes (未使用)")
print(f"K1 GMEM 总流量: {gmem_bytes_k1} bytes")
print(f"K1 AI (GMEM): {ai_k1_gmem:.4f} FLOPs/byte")

print(f"\nK1 结果验证: {'PASS' if np.array_equal(C_k1, C_ref) else 'FAIL'}")

# K1: C[0,0] 完整计算
print("\n--- K1: C[0,0] 完整计算 (thread block(0,0) thread(0,0)) ---")
print(f"C[0,0] = Sum(k=0..15) A[0,k]*B[k,0]")
acc_demo = np.float32(0.0)
for k in range(16):
    prod = A[0, k] * B[k, 0]
    acc_demo += prod
    print(f"  k={k:2d}: {int(A[0,k]):+3d} * {int(B[k,0]):+3d} = {int(prod):+5d}  acc={int(acc_demo):+5d}")
print(f"  C[0,0] = {int(acc_demo):d}  (参考: {int(C_ref[0,0]):d})")

print("\n--- K1: C[0,0]~C[0,3] 验证 ---")
for j in range(4):
    val = np.float32(0.0)
    for k in range(16):
        val += A[0, k] * B[k, j]
    print(f"  C[0,{j}] = {int(val):5d}  (参考: {int(C_ref[0,j]):5d})")

# ============================================================
# 2. K2: SMEM Caching
# ============================================================
print("\n" + "=" * 70)
print("K2: SMEM Caching (共享内存缓存)")
print("BM=BN=BK=4, blockDim=(4,4), gridDim=(4,4)")
print("=" * 70)

C_k2 = np.zeros((16, 16), dtype=np.float32)
gA_k2, gB_k2, gC_k2 = 0, 0, 0

for block_row in range(4):
    for block_col in range(4):
        row_start = block_row * 4
        col_start = block_col * 4
        for kt in range(4):
            k_start = kt * 4
            # Load 4x4 tile from GMEM to SMEM
            As = A[row_start:row_start + 4, k_start:k_start + 4].copy()
            Bs = B[k_start:k_start + 4, col_start:col_start + 4].copy()
            gA_k2 += 16  # GMEM read: 4x4
            gB_k2 += 16  # GMEM read: 4x4
            # Threads read from SMEM
            for ti in range(4):
                for tj in range(4):
                    acc = np.float32(0.0)
                    for kk in range(4):
                        acc += As[ti, kk] * Bs[kk, tj]
                    C_k2[row_start + ti, col_start + tj] += acc
        gC_k2 += 16  # GMEM write: 4x4 per block

gmem_bytes_k2 = (gA_k2 + gB_k2 + gC_k2) * 4
# SMEM accesses: each block reads As(4x4) and Bs(4x4) per kt step, 16 threads each do 4 kk reads * 2 smem reads
# total SMEM reads = 16 blocks * 4 ksteps * (16 threads * 4 kk * 2 smem reads per inner = 128 smem reads per block-kt)
# But more precisely: threads read As[ti,kk] and Bs[kk,tj] each inner loop
smem_reads_k2 = 16 * 4 * 16 * 4 * 2  # 16 blocks * 4 ksteps * 16 threads * 4 kk * 2 smem reads
smem_writes_k2 = 16 * 4 * 2  # each block-kt writes 2 tiles (As, Bs) to smem
smem_bytes_k2 = (smem_reads_k2 + smem_writes_k2 * 16) * 4  # writes are 16-element tiles
ai_k2_gmem = total_flops / gmem_bytes_k2

print(f"\nK2 全局内存 (GMEM) 读取 A: {gA_k2} floats = {gA_k2*4} bytes")
print(f"K2 全局内存 (GMEM) 读取 B: {gB_k2} floats = {gB_k2*4} bytes")
print(f"K2 全局内存 (GMEM) 写入 C: {gC_k2} floats = {gC_k2*4} bytes")
print(f"K2 GMEM 总流量: {gmem_bytes_k2} bytes")
print(f"K2 共享内存 (SMEM) 访问: ~{smem_bytes_k2} bytes (reads + writes)")
print(f"K2 AI (GMEM): {ai_k2_gmem:.4f} FLOPs/byte (K1: {ai_k1_gmem:.4f})")

print(f"\nK2 结果验证: {'PASS' if np.array_equal(C_k2, C_ref) else 'FAIL'}")

# K2: 分步展示
print("\n--- K2: Block(0,0) C[0,0] 分步计算 (K 分 4 步, BK=4) ---")
for kt in range(4):
    k_start = kt * 4
    As = A[0:4, k_start:k_start + 4]
    Bs = B[k_start:k_start + 4, 0:4]
    partial = As @ Bs
    print(f"\n  kt={kt} (k={k_start}..{k_start+3}):")
    print(f"    As = A[0:4, {k_start}:{k_start+4}]")
    for i in range(4):
        print(f"    [{''.join(f'{int(As[i,j]):4d}' for j in range(4))}]")
    print(f"    Bs = B[{k_start}:{k_start+4}, 0:4]")
    for i in range(4):
        print(f"    [{''.join(f'{int(Bs[i,j]):4d}' for j in range(4))}]")
    print(f"    partial C[0:4,0:4] (kt={kt}):")
    for i in range(4):
        print(f"    [{''.join(f'{int(partial[i,j]):5d}' for j in range(4))}]")

print("\n--- K2: C[0,0] 累加 ---")
total_00 = np.float32(0.0)
for kt in range(4):
    k_start = kt * 4
    part = (A[0:4, k_start:k_start+4] @ B[k_start:k_start+4, 0:4])[0, 0]
    total_00 += part
    print(f"  kt={kt}: partial = {int(part):+5d}  running_sum = {int(total_00):+5d}")
print(f"  C[0,0] = {int(total_00):d}  (参考: {int(C_ref[0,0]):d})")

print("\n--- K2: kt=0 共享内存内容 ---")
print("A_smem (4x4):")
for i in range(4):
    print(f"  [{''.join(f'{int(A[i,j]):4d}' for j in range(4))}]")
print("B_smem (4x4):")
for i in range(4):
    print(f"  [{''.join(f'{int(B[i,j]):4d}' for j in range(4))}]")

# ============================================================
# 3. K3: 1D Thread Tile
# ============================================================
print("\n" + "=" * 70)
print("K3: 1D Thread Tile")
print("BM=8, BN=8, BK=4, TM=4, blockDim=(8,2), gridDim=(2,2)")
print("=" * 70)

C_k3 = np.zeros((16, 16), dtype=np.float32)
gA_k3, gB_k3, gC_k3 = 0, 0, 0

for block_row in range(2):
    for block_col in range(2):
        m_start = block_row * 8
        n_start = block_col * 8
        for kt in range(4):
            k_start = kt * 4
            # GMEM -> SMEM: load 8x4 As and 4x8 Bs
            As = A[m_start:m_start + 8, k_start:k_start + 4].copy()
            Bs = B[k_start:k_start + 4, n_start:n_start + 8].copy()
            gA_k3 += 32  # GMEM: 8x4
            gB_k3 += 32  # GMEM: 4x8
            # Threads read from SMEM
            for tx in range(8):
                for ty in range(2):
                    for tm in range(4):
                        row = m_start + ty * 4 + tm
                        acc = np.float32(0.0)
                        for kk in range(4):
                            acc += As[ty * 4 + tm, kk] * Bs[kk, tx]
                        C_k3[row, n_start + tx] += acc
            gC_k3 += 64  # GMEM write per block per k-step

gmem_bytes_k3 = (gA_k3 + gB_k3 + gC_k3) * 4
ai_k3_gmem = total_flops / gmem_bytes_k3

print(f"\nK3 全局内存 (GMEM) 读取 A: {gA_k3} floats = {gA_k3*4} bytes")
print(f"K3 全局内存 (GMEM) 读取 B: {gB_k3} floats = {gB_k3*4} bytes")
print(f"K3 全局内存 (GMEM) 写入 C: {gC_k3} floats = {gC_k3*4} bytes")
print(f"K3 GMEM 总流量: {gmem_bytes_k3} bytes")
print(f"K3 AI (GMEM): {ai_k3_gmem:.4f} FLOPs/byte")

print(f"\nK3 结果验证: {'PASS' if np.array_equal(C_k3, C_ref) else 'FAIL'}")

# K3 寄存器操作
print("\n--- K3: Block(0,0), kt=0, thread(ty=0,tx=0) 寄存器操作 ---")
print("thread(ty=0,tx=0) 计算 C[0:4,0] 的 4 个元素")
As00 = A[0:8, 0:4]
Bs00 = B[0:4, 0:8]
print("\nAs (8x4):")
for i in range(8):
    print(f"  [{''.join(f'{int(As00[i,j]):4d}' for j in range(4))}]")
print("\nBs (4x8) 列0..7:")
for i in range(4):
    print(f"  [{''.join(f'{int(Bs00[i,j]):4d}' for j in range(8))}]")

print("\nThread(ty=0,tx=0) reg_A[0..3][*] = As[0..3, 0..4]:")
for t in range(4):
    print(f"  reg_A[{t}] = [{', '.join(f'{int(As00[t,j]):3d}' for j in range(4))}]")
print("Thread(ty=0,tx=0) reg_B[*][0] = Bs[:,0]:")
for kk in range(4):
    print(f"  reg_B[{kk}] = {int(Bs00[kk, 0]):3d}")

print("\nFMA 计算 (kt=0 partial):")
for tm in range(4):
    acc = np.float32(0.0)
    for kk in range(4):
        prod = As00[tm, kk] * Bs00[kk, 0]
        acc += prod
        print(f"  tm={tm}, A[{tm},{kk}]*B[{kk},0] = {int(As00[tm,kk]):+3d}*{int(Bs00[kk,0]):+3d} = {int(prod):+5d}")
    print(f"  => partial C[{tm},0] (kt=0) = {int(acc):d}")

# ============================================================
# 4. K4: 2D Thread Tile
# ============================================================
print("\n" + "=" * 70)
print("K4: 2D Thread Tile")
print("BM=8, BN=8, BK=4, TM=4, TN=4, blockDim=(2,2), gridDim=(2,2)")
print("=" * 70)

C_k4 = np.zeros((16, 16), dtype=np.float32)
gA_k4, gB_k4, gC_k4 = 0, 0, 0

for block_row in range(2):
    for block_col in range(2):
        m_start = block_row * 8
        n_start = block_col * 8
        for kt in range(4):
            k_start = kt * 4
            # GMEM -> SMEM
            As = A[m_start:m_start + 8, k_start:k_start + 4].copy()
            Bs = B[k_start:k_start + 4, n_start:n_start + 8].copy()
            gA_k4 += 32
            gB_k4 += 32
            for tx in range(2):
                for ty in range(2):
                    for tm in range(4):
                        for tn in range(4):
                            row = m_start + ty * 4 + tm
                            col = n_start + tx * 4 + tn
                            acc = np.float32(0.0)
                            for kk in range(4):
                                acc += As[ty * 4 + tm, kk] * Bs[kk, tx * 4 + tn]
                            C_k4[row, col] += acc
            gC_k4 += 64

gmem_bytes_k4 = (gA_k4 + gB_k4 + gC_k4) * 4
ai_k4_gmem = total_flops / gmem_bytes_k4

print(f"\nK4 全局内存 (GMEM) 读取 A: {gA_k4} floats = {gA_k4*4} bytes")
print(f"K4 全局内存 (GMEM) 读取 B: {gB_k4} floats = {gB_k4*4} bytes")
print(f"K4 全局内存 (GMEM) 写入 C: {gC_k4} floats = {gC_k4*4} bytes")
print(f"K4 GMEM 总流量: {gmem_bytes_k4} bytes")
print(f"K4 AI (GMEM): {ai_k4_gmem:.4f} FLOPs/byte")

print(f"\nK4 结果验证: {'PASS' if np.array_equal(C_k4, C_ref) else 'FAIL'}")

# K4 详细
print("\n--- K4: Block(0,0) 线程 2D 职责 ---")
print("Block(0,0) 有 blockDim=(2,2)=4 线程, 各计算 4x4=16 个 C 元素")
print("\nThread(ty=0,tx=0): C[0:4, 0:4]")
print("Thread(ty=0,tx=1): C[0:4, 4:8]")
print("Thread(ty=1,tx=0): C[4:8, 0:4]")
print("Thread(ty=1,tx=1): C[4:8, 4:8]")

print("\n--- K4: kt=0, thread(ty=0,tx=0) reg_A (4x4) ---")
for i in range(4):
    print(f"  [{''.join(f'{int(A[i,j]):4d}' for j in range(4))}]")
print("--- K4: kt=0, thread(ty=0,tx=0) reg_B (4x4) ---")
for i in range(4):
    print(f"  [{''.join(f'{int(B[i,j]):4d}' for j in range(4))}]")

partial_4x4 = A[0:4, 0:4] @ B[0:4, 0:4]
print("--- K4: kt=0, thread(0,0) partial 4x4 ---")
for i in range(4):
    print(f"  [{''.join(f'{int(partial_4x4[i,j]):5d}' for j in range(4))}]")

# ============================================================
# 5. K5: Register Caching
# ============================================================
print("\n" + "=" * 70)
print("K5: Register Caching (寄存器缓存)")
print("同 K4 参数, 显式 reg_A[TM][BK], reg_B[BK][TN]")
print("=" * 70)

C_k5 = np.zeros((16, 16), dtype=np.float32)
gA_k5, gB_k5, gC_k5 = 0, 0, 0

for block_row in range(2):
    for block_col in range(2):
        m_start = block_row * 8
        n_start = block_col * 8
        for kt in range(4):
            k_start = kt * 4
            As = A[m_start:m_start + 8, k_start:k_start + 4].copy()
            Bs = B[k_start:k_start + 4, n_start:n_start + 8].copy()
            gA_k5 += 32
            gB_k5 += 32
            for tx in range(2):
                for ty in range(2):
                    reg_A = As[ty * 4:ty * 4 + 4, :].copy()
                    reg_B = Bs[:, tx * 4:tx * 4 + 4].copy()
                    for tm in range(4):
                        for tn in range(4):
                            acc = np.float32(0.0)
                            for kk in range(4):
                                acc += reg_A[tm, kk] * reg_B[kk, tn]
                            C_k5[m_start + ty * 4 + tm, n_start + tx * 4 + tn] += acc
            gC_k5 += 64

gmem_bytes_k5 = (gA_k5 + gB_k5 + gC_k5) * 4
ai_k5_gmem = total_flops / gmem_bytes_k5

print(f"\nK5 全局内存 (GMEM) 读取 A: {gA_k5} floats = {gA_k5*4} bytes")
print(f"K5 全局内存 (GMEM) 读取 B: {gB_k5} floats = {gB_k5*4} bytes")
print(f"K5 全局内存 (GMEM) 写入 C: {gC_k5} floats = {gC_k5*4} bytes")
print(f"K5 GMEM 总流量: {gmem_bytes_k5} bytes")
print(f"K5 AI (GMEM): {ai_k5_gmem:.4f} FLOPs/byte")

print(f"\nK5 结果验证: {'PASS' if np.array_equal(C_k5, C_ref) else 'FAIL'}")

# ============================================================
# 6. K6: Float4 Vectorized
# ============================================================
print("\n" + "=" * 70)
print("K6: Float4 Vectorized (float4 向量化加载)")
print("同 K5, 每次加载 128-bit = 4 floats")
print("=" * 70)

C_k6 = np.zeros((16, 16), dtype=np.float32)
gA_k6, gB_k6, gC_k6 = 0, 0, 0

for block_row in range(2):
    for block_col in range(2):
        m_start = block_row * 8
        n_start = block_col * 8
        for kt in range(4):
            k_start = kt * 4
            As = A[m_start:m_start + 8, k_start:k_start + 4].copy()
            Bs = B[k_start:k_start + 4, n_start:n_start + 8].copy()
            gA_k6 += 8   # 32 floats / 4 = 8 float4 transactions
            gB_k6 += 8
            for tx in range(2):
                for ty in range(2):
                    reg_A = As[ty * 4:ty * 4 + 4, :].copy()
                    reg_B = Bs[:, tx * 4:tx * 4 + 4].copy()
                    for tm in range(4):
                        for tn in range(4):
                            acc = np.float32(0.0)
                            for kk in range(4):
                                acc += reg_A[tm, kk] * reg_B[kk, tn]
                            C_k6[m_start + ty * 4 + tm, n_start + tx * 4 + tn] += acc
            gC_k6 += 64

gmem_bytes_k6 = (gA_k6 * 4 + gB_k6 * 4 + gC_k6) * 4
ai_k6_gmem = total_flops / gmem_bytes_k6

print(f"\nK6 float4 事务读取 A: {gA_k6} 次 (= 32 floats / 4 per kt per block)")
print(f"K6 float4 事务读取 B: {gB_k6} 次")
print(f"K6 等效 float 读取: A={gA_k6*4}, B={gB_k6*4}")
print(f"K6 全局内存 (GMEM) 流量: {gmem_bytes_k6} bytes")
print(f"K6 AI (GMEM): {ai_k6_gmem:.4f} FLOPs/byte (与 K5 同数据量, 提高总线利用率)")

print(f"\nK6 结果验证: {'PASS' if np.array_equal(C_k6, C_ref) else 'FAIL'}")

print("\n--- K6 Float4 原理 ---")
print("K5: 4 次 32-bit 加载事务")
print("K6: 1 次 128-bit 加载事务 (float4)")
print(f"As[0, 0:4] = float4({int(A[0,0]):d}, {int(A[0,1]):d}, {int(A[0,2]):d}, {int(A[0,3]):d})")
print(f"As[1, 0:4] = float4({int(A[1,0]):d}, {int(A[1,1]):d}, {int(A[1,2]):d}, {int(A[1,3]):d})")

# ============================================================
# 7. K7: Double Buffering
# ============================================================
print("\n" + "=" * 70)
print("K7: Double Buffering (双缓冲)")
print("同 K6, ping-pong 共享内存缓冲")
print("=" * 70)

C_k7 = np.zeros((16, 16), dtype=np.float32)
gA_k7, gB_k7, gC_k7 = 0, 0, 0

for block_row in range(2):
    for block_col in range(2):
        m_start = block_row * 8
        n_start = block_col * 8

        # 预加载 kt=0
        cur_As = A[m_start:m_start + 8, 0:4].copy()
        cur_Bs = B[0:4, n_start:n_start + 8].copy()
        gA_k7 += 8
        gB_k7 += 8

        for kt in range(1, 4):  # kt=1,2,3
            k_next = kt * 4
            next_As = A[m_start:m_start + 8, k_next:k_next + 4].copy()
            next_Bs = B[k_next:k_next + 4, n_start:n_start + 8].copy()
            gA_k7 += 8
            gB_k7 += 8

            # 用 cur_As/cur_Bs 计算, 同时 next_As/next_Bs 在"加载中"
            for tx in range(2):
                for ty in range(2):
                    reg_A = cur_As[ty * 4:ty * 4 + 4, :].copy()
                    reg_B = cur_Bs[:, tx * 4:tx * 4 + 4].copy()
                    for tm in range(4):
                        for tn in range(4):
                            acc = np.float32(0.0)
                            for kk in range(4):
                                acc += reg_A[tm, kk] * reg_B[kk, tn]
                            C_k7[m_start + ty * 4 + tm, n_start + tx * 4 + tn] += acc
            cur_As, cur_Bs = next_As, next_Bs

        # 最后一个 tile (kt=3)
        for tx in range(2):
            for ty in range(2):
                reg_A = cur_As[ty * 4:ty * 4 + 4, :].copy()
                reg_B = cur_Bs[:, tx * 4:tx * 4 + 4].copy()
                for tm in range(4):
                    for tn in range(4):
                        acc = np.float32(0.0)
                        for kk in range(4):
                            acc += reg_A[tm, kk] * reg_B[kk, tn]
                        C_k7[m_start + ty * 4 + tm, n_start + tx * 4 + tn] += acc

        gC_k7 += 64

gmem_bytes_k7 = (gA_k7 * 4 + gB_k7 * 4 + gC_k7) * 4
ai_k7_gmem = total_flops / gmem_bytes_k7

print(f"\nK7 float4 事务读取 A: {gA_k7} 次")
print(f"K7 float4 事务读取 B: {gB_k7} 次")
print(f"K7 全局内存 (GMEM) 流量: {gmem_bytes_k7} bytes")
print(f"K7 AI (GMEM): {ai_k7_gmem:.4f} FLOPs/byte")

print(f"\nK7 结果验证: {'PASS' if np.array_equal(C_k7, C_ref) else 'FAIL'}")

print("\n--- K7 Ping-Pong 时间线 (Block(0,0)) ---")
print("""
  kt=0: [LOAD (k=0..3)  -> PING]         (预加载)
         __syncthreads()
  kt=1: [LOAD (k=4..7)  -> PONG]   [COMPUTE with PING]
         __syncthreads()
  kt=2: [LOAD (k=8..11) -> PING]   [COMPUTE with PONG]
         __syncthreads()
  kt=3:                              [COMPUTE with PING]
         __syncthreads()
""")

# ============================================================
# 8. 汇总对比
# ============================================================
print("\n" + "=" * 70)
print("汇总对比")
print("=" * 70)
print(f"{'Kernel':<14} {'GMEM bytes':>12} {'SMEM bytes':>12} {'AI (GMEM)':>12} {'AI (SMEM)':>12} {'vs K1':>8}")
print("-" * 70)

kernels = [
    ("K1 Naive",     gmem_bytes_k1,   0,                ai_k1_gmem, 0.0),
    ("K2 SMEM",      gmem_bytes_k2,   smem_bytes_k2,    ai_k2_gmem, 0.0),
    ("K3 1D Tile",   gmem_bytes_k3,   0,                ai_k3_gmem, 0.0),
    ("K4 2D Tile",   gmem_bytes_k4,   0,                ai_k4_gmem, 0.0),
    ("K5 Reg Cache", gmem_bytes_k5,   0,                ai_k5_gmem, 0.0),
    ("K6 Float4",    gmem_bytes_k6,   0,                ai_k6_gmem, 0.0),
    ("K7 DblBuf",    gmem_bytes_k7,   0,                ai_k7_gmem, 0.0),
]

for name, gmem, smem, ai_g, ai_s in kernels:
    ratio = ai_g / ai_k1_gmem
    print(f"{name:<14} {gmem:>12} {smem:>12} {ai_g:>12.4f} {ai_s:>12.4f} {ratio:>7.2f}x")

# ============================================================
# 9. 最终验证
# ============================================================
print("\n" + "=" * 70)
print("最终验证: 所有 Kernel 结果 vs numpy A@B")
print("=" * 70)
results = [
    ("K1 Naive", C_k1),
    ("K2 SMEM", C_k2),
    ("K3 1D Tile", C_k3),
    ("K4 2D Tile", C_k4),
    ("K5 Reg Cache", C_k5),
    ("K6 Float4", C_k6),
    ("K7 DblBuf", C_k7),
]
all_pass = True
for name, C_k in results:
    ok = np.array_equal(C_k, C_ref)
    if not ok:
        all_pass = False
    print(f"  {name:<14}: {'PASS' if ok else 'FAIL'}")

print(f"\n全部通过: {'YES' if all_pass else 'NO - see above'}")
print("=" * 70)
print("验证完成")
print("=" * 70)

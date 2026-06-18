#!/usr/bin/env python3
"""
CUDA SGEMM 优化验证脚本
基于 wangzyon/NVIDIA_SGEMM_PRACTICE
演示问题规模: M=N=K=16, seed=2024, float32

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

np.random.seed(2024)

A = np.random.randn(16, 16).astype(np.float32)
B = np.random.randn(16, 16).astype(np.float32)
C_ref = A @ B

print("=" * 70)
print("CUDA SGEMM 优化验证")
print("问题规模: M=N=K=16, seed=2024, dtype=float32")
print("=" * 70)

# ---- 矩阵 A ----
print("\n>>> 矩阵 A (16x16):")
print("     ", end="")
for j in range(16):
    print(f"{'col':>4s}{j:<4d}", end="")
print()
for i in range(16):
    print(f"row{i:2d} ", end="")
    for j in range(16):
        print(f"{A[i, j]:10.6f}", end="")
    print()

# ---- 矩阵 B ----
print("\n>>> 矩阵 B (16x16):")
print("     ", end="")
for j in range(16):
    print(f"{'col':>4s}{j:<4d}", end="")
print()
for i in range(16):
    print(f"row{i:2d} ", end="")
    for j in range(16):
        print(f"{B[i, j]:10.6f}", end="")
    print()

# ---- 参考结果 C ----
print("\n>>> 参考结果 C = A@B (16x16):")
print("     ", end="")
for j in range(16):
    print(f"{'col':>4s}{j:<4d}", end="")
print()
for i in range(16):
    print(f"row{i:2d} ", end="")
    for j in range(16):
        print(f"{C_ref[i, j]:10.6f}", end="")
    print()

print(f"\nC[0,0]={C_ref[0,0]:.6f}  C[0,15]={C_ref[0,15]:.6f}")
print(f"C[15,0]={C_ref[15,0]:.6f}  C[15,15]={C_ref[15,15]:.6f}")

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
                    gA_k1 += 1
                    gB_k1 += 1
                C_k1[row, col] = acc
                gC_k1 += 1

print(f"\nK1 全局内存读取 A: {gA_k1} 次")
print(f"K1 全局内存读取 B: {gB_k1} 次")
print(f"K1 全局内存写入 C: {gC_k1} 次")

total_fmas = 16 * 16 * 16
total_flops = total_fmas * 2
ai_k1 = total_flops / ((gA_k1 + gB_k1 + gC_k1) * 4)
print(f"K1 总 FMA 数: {total_fmas} (= M*N*K)")
print(f"K1 总 FLOPs: {total_flops} (= FMA*2)")
print(f"K1 全局内存流量: {(gA_k1+gB_k1+gC_k1)*4} bytes")
print(f"K1 计算访存比: {ai_k1:.3f} FLOPs/byte")

print(f"\nK1 结果验证: {'PASS' if np.allclose(C_k1, C_ref, rtol=1e-4) else 'FAIL'}")
print(f"K1 最大误差: {np.abs(C_k1 - C_ref).max():.6e}")

# K1: C[0,0] 完整计算
print("\n--- K1: C[0,0] 完整计算 (thread block(0,0) thread(0,0)) ---")
print(f"C[0,0] = Sum(k=0..15) A[0,k]*B[k,0]")
acc_demo = np.float32(0.0)
for k in range(16):
    prod = A[0, k] * B[k, 0]
    acc_demo += prod
    print(f"  k={k:2d}: {A[0,k]:+10.6f} * {B[k,0]:+10.6f} = {prod:+12.8f}  acc={acc_demo:+12.8f}")
print(f"  C[0,0] = {acc_demo:.6f}  (参考: {C_ref[0,0]:.6f})")

print("\n--- K1: C[0,0]~C[0,3] 验证 ---")
for j in range(4):
    val = np.float32(0.0)
    for k in range(16):
        val += A[0, k] * B[k, j]
    print(f"  C[0,{j}] = {val:.6f}  (参考: {C_ref[0,j]:.6f})")

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
            As = A[row_start:row_start + 4, k_start:k_start + 4].copy()
            Bs = B[k_start:k_start + 4, col_start:col_start + 4].copy()
            gA_k2 += 16
            gB_k2 += 16
            for ti in range(4):
                for tj in range(4):
                    acc = np.float32(0.0)
                    for kk in range(4):
                        acc += As[ti, kk] * Bs[kk, tj]
                    C_k2[row_start + ti, col_start + tj] += acc
        gC_k2 += 16  # 4x4 elements per block

print(f"\nK2 全局内存读取 A: {gA_k2} 次 (=16 blocks * 16 elems * 4 ksteps)")
print(f"K2 全局内存读取 B: {gB_k2} 次")
print(f"K2 全局内存写入 C: {gC_k2} 次")

ai_k2 = total_flops / ((gA_k2 + gB_k2 + gC_k2) * 4)
print(f"K2 全局内存流量: {(gA_k2+gB_k2+gC_k2)*4} bytes")
print(f"K2 计算访存比: {ai_k2:.3f} FLOPs/byte (K1: {ai_k1:.3f})")

print(f"\nK2 结果验证: {'PASS' if np.allclose(C_k2, C_ref, rtol=1e-4) else 'FAIL'}")
print(f"K2 最大误差: {np.abs(C_k2 - C_ref).max():.6e}")

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
        print(f"    [{''.join(f'{As[i,j]:10.6f}' for j in range(4))}]")
    print(f"    Bs = B[{k_start}:{k_start+4}, 0:4]")
    for i in range(4):
        print(f"    [{''.join(f'{Bs[i,j]:10.6f}' for j in range(4))}]")
    print(f"    partial C[0:4,0:4] (kt={kt}):")
    for i in range(4):
        print(f"    [{''.join(f'{partial[i,j]:10.6f}' for j in range(4))}]")

print("\n--- K2: C[0,0] 累加 ---")
total_00 = np.float32(0.0)
for kt in range(4):
    k_start = kt * 4
    part = (A[0:4, k_start:k_start+4] @ B[k_start:k_start+4, 0:4])[0, 0]
    total_00 += part
    print(f"  kt={kt}: partial = {part:+12.8f}  running_sum = {total_00:+12.8f}")
print(f"  C[0,0] = {total_00:.6f}  (参考: {C_ref[0,0]:.6f})")

# K2 共享内存内容
print("\n--- K2: kt=0 共享内存内容 ---")
print("A_smem (4x4):")
for i in range(4):
    print(f"  [{''.join(f'{A[i,j]:10.6f}' for j in range(4))}]")
print("B_smem (4x4):")
for i in range(4):
    print(f"  [{''.join(f'{B[i,j]:10.6f}' for j in range(4))}]")

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
            As = A[m_start:m_start + 8, k_start:k_start + 4].copy()
            Bs = B[k_start:k_start + 4, n_start:n_start + 8].copy()
            gA_k3 += 32
            gB_k3 += 32
            for tx in range(8):
                for ty in range(2):
                    for tm in range(4):
                        row = m_start + ty * 4 + tm
                        acc = np.float32(0.0)
                        for kk in range(4):
                            acc += As[ty * 4 + tm, kk] * Bs[kk, tx]
                        C_k3[row, n_start + tx] += acc
            gC_k3 += 64  # 8x8 per block per k-step, but final write counts once

print(f"\nK3 全局内存读取 A: {gA_k3} 次 (=4 blocks * 4 ksteps * 32)")
print(f"K3 全局内存读取 B: {gB_k3} 次")
print(f"K3 全局内存写入 C: {gC_k3} 次")

ai_k3 = total_flops / ((gA_k3 + gB_k3 + gC_k3) * 4)
print(f"K3 全局内存流量: {(gA_k3+gB_k3+gC_k3)*4} bytes")
print(f"K3 计算访存比: {ai_k3:.3f} FLOPs/byte")

print(f"\nK3 结果验证: {'PASS' if np.allclose(C_k3, C_ref, rtol=1e-4) else 'FAIL'}")
print(f"K3 最大误差: {np.abs(C_k3 - C_ref).max():.6e}")

# K3 寄存器操作
print("\n--- K3: Block(0,0), kt=0, thread(ty=0,tx=0) 寄存器操作 ---")
print("thread(ty=0,tx=0) 计算 C[0:4,0] 的 4 个元素")
As00 = A[0:8, 0:4]
Bs00 = B[0:4, 0:8]
print("\nAs (8x4):")
for i in range(8):
    print(f"  [{''.join(f'{As00[i,j]:10.6f}' for j in range(4))}]")
print("\nBs (4x8) 列0..7:")
for i in range(4):
    print(f"  [{''.join(f'{Bs00[i,j]:10.6f}' for j in range(8))}]")

print("\nThread(ty=0,tx=0) reg_A[0..3][*] = As[0..3, 0..4]:")
for t in range(4):
    print(f"  reg_A[{t}] = [{''.join(f'{As00[t,j]:10.6f}' for j in range(4))}]")
print("Thread(ty=0,tx=0) reg_B[*][0] = Bs[:,0]:")
for kk in range(4):
    print(f"  reg_B[{kk}] = {Bs00[kk, 0]:10.6f}")

print("\nFMA 计算 (kt=0 partial):")
for tm in range(4):
    acc = np.float32(0.0)
    for kk in range(4):
        prod = As00[tm, kk] * Bs00[kk, 0]
        acc += prod
        print(f"  tm={tm}, A[{tm},{kk}]*B[{kk},0] = {As00[tm,kk]:+10.6f}*{Bs00[kk,0]:+10.6f} = {prod:+12.8f}")
    print(f"  => partial C[{tm},0] (kt=0) = {acc:.6f}")

print("\n--- K3: Block(0,0) 线程映射 ---")
print("blockDim=(8,2): tx(0..7 N方向), ty(0..1 M方向每线程TM=4行)")
print("Block 覆盖 C[0:8,0:8] = 64 元素")
print("Thread(ty=0,tx=0): C[0,0],C[1,0],C[2,0],C[3,0]")
print("Thread(ty=1,tx=0): C[4,0],C[5,0],C[6,0],C[7,0]")
print("Thread(ty=0,tx=1): C[0,1],C[1,1],C[2,1],C[3,1]")

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

print(f"\nK4 全局内存读取 A: {gA_k4} 次")
print(f"K4 全局内存读取 B: {gB_k4} 次")
print(f"K4 全局内存写入 C: {gC_k4} 次")

ai_k4 = total_flops / ((gA_k4 + gB_k4 + gC_k4) * 4)
print(f"K4 全局内存流量: {(gA_k4+gB_k4+gC_k4)*4} bytes")
print(f"K4 计算访存比: {ai_k4:.3f} FLOPs/byte")

print(f"\nK4 结果验证: {'PASS' if np.allclose(C_k4, C_ref, rtol=1e-4) else 'FAIL'}")
print(f"K4 最大误差: {np.abs(C_k4 - C_ref).max():.6e}")

# K4 详细
print("\n--- K4: Block(0,0) 线程 2D 职责 ---")
print("Block(0,0) 有 blockDim=(2,2)=4 线程, 各计算 4x4=16 个 C 元素")
print("Block 覆盖 C[0:8,0:8] = 64 元素 = 4 threads * 16")
print("\nThread(ty=0,tx=0): C[0:4, 0:4]")
print("Thread(ty=0,tx=1): C[0:4, 4:8]")
print("Thread(ty=1,tx=0): C[4:8, 0:4]")
print("Thread(ty=1,tx=1): C[4:8, 4:8]")

print("\n--- K4: kt=0, thread(ty=0,tx=0) reg_A (4x4) ---")
for i in range(4):
    print(f"  [{''.join(f'{A[i,j]:10.6f}' for j in range(4))}]")
print("--- K4: kt=0, thread(ty=0,tx=0) reg_B (4x4) ---")
for i in range(4):
    print(f"  [{''.join(f'{B[i,j]:10.6f}' for j in range(4))}]")

partial_4x4 = A[0:4, 0:4] @ B[0:4, 0:4]
print("--- K4: kt=0, thread(0,0) partial 4x4 ---")
for i in range(4):
    print(f"  [{''.join(f'{partial_4x4[i,j]:10.6f}' for j in range(4))}]")

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

ai_k5 = total_flops / ((gA_k5 + gB_k5 + gC_k5) * 4)
print(f"\nK5 全局内存读取 A: {gA_k5} 次")
print(f"K5 全局内存读取 B: {gB_k5} 次")
print(f"K5 全局内存流量: {(gA_k5+gB_k5+gC_k5)*4} bytes")
print(f"K5 计算访存比: {ai_k5:.3f} FLOPs/byte")

print(f"\nK5 结果验证: {'PASS' if np.allclose(C_k5, C_ref, rtol=1e-4) else 'FAIL'}")
print(f"K5 最大误差: {np.abs(C_k5 - C_ref).max():.6e}")

print("\n--- K5 原理 ---")
print("K4: FMA 每次从共享内存读取 (共享内存延迟 ~20-30 cycles)")
print("K5: 先将 As/Bs 片段加载到寄存器, 再执行 FMA")
print("  展开: reg_A[0][0]*reg_B[0][0] + reg_A[0][1]*reg_B[1][0] + ...")
print("  float reg_A[TM][BK] = reg_A[4][4] = 16 regs")
print("  float reg_B[BK][TN] = reg_B[4][4] = 16 regs")
print("  计算 4x4 FMA 全在寄存器中完成")

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

ai_k6 = total_flops / ((gA_k6 * 4 + gB_k6 * 4 + gC_k6) * 4)
print(f"\nK6 float4 事务读取 A: {gA_k6} 次 (= 32 floats / 4 per kt per block)")
print(f"K6 float4 事务读取 B: {gB_k6} 次")
print(f"K6 等效 float 读取: A={gA_k6*4}, B={gB_k6*4}")
print(f"K6 全局内存流量: {(gA_k6*4+gB_k6*4+gC_k6)*4} bytes")
print(f"K6 计算访存比: {ai_k6:.3f} FLOPs/byte (与 K5 同数据量, 提高总线利用率)")

print(f"\nK6 结果验证: {'PASS' if np.allclose(C_k6, C_ref, rtol=1e-4) else 'FAIL'}")
print(f"K6 最大误差: {np.abs(C_k6 - C_ref).max():.6e}")

print("\n--- K6 Float4 原理 ---")
print("K5: 4 次 32-bit 加载事务")
print("K6: 1 次 128-bit 加载事务 (float4)")
print(f"As[0, 0:4] = float4({A[0,0]:.6f}, {A[0,1]:.6f}, {A[0,2]:.6f}, {A[0,3]:.6f})")
print(f"As[1, 0:4] = float4({A[1,0]:.6f}, {A[1,1]:.6f}, {A[1,2]:.6f}, {A[1,3]:.6f})")

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

ai_k7 = total_flops / ((gA_k7 * 4 + gB_k7 * 4 + gC_k7) * 4)
print(f"\nK7 float4 事务读取 A: {gA_k7} 次")
print(f"K7 float4 事务读取 B: {gB_k7} 次")
print(f"K7 全局内存流量: {(gA_k7*4+gB_k7*4+gC_k7)*4} bytes")
print(f"K7 计算访存比: {ai_k7:.3f} FLOPs/byte")

print(f"\nK7 结果验证: {'PASS' if np.allclose(C_k7, C_ref, rtol=1e-4) else 'FAIL'}")
print(f"K7 最大误差: {np.abs(C_k7 - C_ref).max():.6e}")

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
         (no more K steps)           [COMPUTE with PONG 如果有]
""")

# ============================================================
# 8. 汇总对比
# ============================================================
print("\n" + "=" * 70)
print("汇总对比")
print("=" * 70)
print(f"{'Kernel':<14} {'Global Mem Bytes':>18} {'Arith Intensity':>18} {'AI vs K1':>10}")
print("-" * 65)

kernels = [
    ("K1 Naive",     gA_k1 + gB_k1 + gC_k1, ai_k1),
    ("K2 SMEM",      gA_k2 + gB_k2 + gC_k2, ai_k2),
    ("K3 1D Tile",   gA_k3 + gB_k3 + gC_k3, ai_k3),
    ("K4 2D Tile",   gA_k4 + gB_k4 + gC_k4, ai_k4),
    ("K5 Reg Cache", gA_k5 + gB_k5 + gC_k5, ai_k5),
    ("K6 Float4",    gA_k6 * 4 + gB_k6 * 4 + gC_k6, ai_k6),
    ("K7 DblBuf",    gA_k7 * 4 + gB_k7 * 4 + gC_k7, ai_k7),
]

for name, mem_count, ai_val in kernels:
    mem_bytes = mem_count * 4
    ratio = ai_val / ai_k1
    print(f"{name:<14} {mem_bytes:>18} {ai_val:>18.3f} {ratio:>9.2f}x")

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
    ok = np.allclose(C_k, C_ref, rtol=1e-4)
    if not ok:
        all_pass = False
    max_diff = np.abs(C_k - C_ref).max()
    print(f"  {name:<14}: {'PASS' if ok else 'FAIL'} (max diff = {max_diff:.2e})")

print(f"\n全部通过: {'YES' if all_pass else 'NO - see above'}")
print("=" * 70)
print("验证完成")
print("=" * 70)

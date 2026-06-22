# 深入理解 CUDA 矩阵乘法：从 Naive 到接近 cuBLAS 的手把手推导

> 基于 [siboehm 的工作日志](https://siboehm.com/articles/22/CUDA-MMM)，使用 M=N=K=16 的具体数值（0-9 整数矩阵, seed=1），逐步推演每一步优化的原理与计算过程。

---

## 目录

1. [演示参数与矩阵](#1-演示参数与矩阵)
2. [Kernel 1：Naive 实现](#2-kernel-1naive-实现)
3. [Kernel 2：全局内存合并访问](#3-kernel-2全局内存合并访问)
4. [Kernel 3：共享内存缓存分块](#4-kernel-3共享内存缓存分块)
5. [Kernel 4：1D Blocktiling](#5-kernel-41d-blocktiling)
6. [Kernel 5：2D Blocktiling — 提高算术强度](#6-kernel-52d-blocktiling--提高算术强度)
7. [Kernel 6：向量化内存访问](#7-kernel-6向量化内存访问)
8. [Kernel 10：Warp 级分块](#8-kernel-10warp-级分块)
9. [性能分析：Occupancy、算术强度与 Roofline 模型](#9-性能分析occupancy算术强度与-roofline-模型)
10. [总结](#10-总结)

---

## 1. 演示参数与矩阵

### 问题规模

| 参数 | 值 |
|------|-----|
| M (A的行数) | 16 |
| N (B的列数) | 16 |
| K (内积维度) | 16 |
| 数据类型 | float32 |
| 随机种子 | 1 (randint 0..9) |
| 矩阵特征 | 0-9 整数矩阵，便于手算验证 |

### 矩阵 A (16×16, float32)

| 行 | 列 0-7 | 列 8-15 |
|----|--------|---------|
| 0 | 5, 8, 9, 5, 0, 0, 1, 7 | 6, 9, 2, 4, 5, 2, 4, 2 |
| 1 | 4, 7, 7, 9, 1, 7, 0, 6 | 9, 9, 7, 6, 9, 1, 0, 1 |
| 2 | 8, 8, 3, 9, 8, 7, 3, 6 | 5, 1, 9, 3, 4, 8, 1, 4 |
| 3 | 0, 3, 9, 2, 0, 4, 9, 2 | 7, 7, 9, 8, 6, 9, 3, 7 |
| 4 | 7, 4, 5, 9, 3, 6, 8, 0 | 2, 7, 7, 9, 7, 3, 0, 8 |
| 5 | 7, 7, 1, 1, 3, 0, 8, 6 | 4, 5, 6, 2, 5, 7, 8, 4 |
| 6 | 4, 7, 7, 4, 9, 0, 2, 0 | 7, 1, 7, 9, 8, 4, 0, 1 |
| 7 | 9, 8, 2, 3, 1, 2, 7, 2 | 6, 0, 9, 2, 6, 6, 2, 7 |
| 8 | 7, 0, 6, 5, 1, 4, 6, 0 | 6, 5, 1, 2, 1, 5, 4, 0 |
| 9 | 7, 8, 9, 5, 7, 0, 9, 3 | 9, 1, 4, 4, 6, 8, 8, 9 |
| 10 | 2, 7, 5, 5, 4, 5, 8, 5 | 8, 1, 1, 8, 7, 0, 3, 4 |
| 11 | 2, 0, 3, 5, 1, 2, 4, 3 | 0, 6, 0, 7, 2, 8, 3, 0 |
| 12 | 8, 4, 2, 9, 0, 3, 8, 1 | 4, 3, 3, 6, 7, 3, 5, 3 |
| 13 | 2, 4, 4, 0, 3, 3, 8, 3 | 5, 6, 7, 5, 1, 7, 0, 2 |
| 14 | 8, 2, 1, 4, 0, 4, 1, 7 | 3, 1, 6, 6, 9, 6, 9, 6 |
| 15 | 0, 0, 2, 9, 6, 0, 6, 7 | 0, 3, 9, 0, 3, 4, 7, 5 |

### 矩阵 B (16×16, float32)

| 行 | 列 0-7 | 列 8-15 |
|----|--------|---------|
| 0 | 3, 8, 8, 0, 6, 7, 9, 5 | 4, 9, 5, 2, 5, 6, 6, 8 |
| 1 | 7, 7, 7, 2, 6, 0, 5, 2 | 1, 8, 5, 9, 4, 9, 1, 2 |
| 2 | 0, 4, 7, 0, 6, 2, 4, 3 | 6, 7, 6, 3, 0, 6, 4, 7 |
| 3 | 6, 2, 9, 5, 9, 9, 9, 8 | 6, 4, 2, 9, 4, 0, 0, 3 |
| 4 | 4, 9, 3, 9, 1, 2, 5, 4 | 0, 8, 2, 3, 9, 9, 4, 4 |
| 5 | 8, 2, 1, 6, 3, 8, 9, 7 | 0, 5, 2, 2, 8, 5, 0, 5 |
| 6 | 9, 8, 6, 6, 0, 4, 7, 3 | 0, 1, 6, 0, 6, 1, 6, 4 |
| 7 | 2, 5, 4, 6, 2, 9, 2, 7 | 5, 0, 7, 8, 8, 8, 0, 7 |
| 8 | 2, 0, 7, 1, 1, 9, 5, 1 | 5, 9, 6, 4, 9, 8, 7, 5 |
| 9 | 1, 8, 0, 5, 3, 9, 0, 4 | 8, 6, 2, 4, 3, 2, 0, 0 |
| 10 | 4, 2, 5, 0, 0, 3, 8, 5 | 3, 1, 4, 7, 3, 2, 2, 2 |
| 11 | 6, 6, 0, 1, 5, 6, 5, 8 | 8, 5, 5, 7, 5, 9, 1, 3 |
| 12 | 9, 3, 3, 3, 6, 1, 3, 0 | 5, 0, 5, 2, 7, 6, 4, 0 |
| 13 | 2, 4, 8, 7, 6, 7, 7, 1 | 7, 7, 3, 8, 3, 0, 6, 3 |
| 14 | 0, 6, 5, 9, 6, 4, 6, 6 | 2, 2, 4, 1, 2, 3, 9, 3 |
| 15 | 6, 7, 0, 3, 3, 6, 8, 6 | 5, 1, 3, 2, 6, 3, 6, 7 |

### 预期结果 C = A @ B (16×16, float32) — Ground Truth

| 行 | 列 0-7 | 列 8-15 |
|----|--------|---------|
| 0 | 238, 346, 341, 215, 316, 377, 322, 282 | 344, 347, 319, 346, 307, 367, 210, 265 |
| 1 | 367, 340, 378, 243, 352, 476, 433, 357 | 392, 400, 354, 429, 428, 436, 194, 290 |
| 2 | 402, 412, 456, 331, 349, 468, 559, 385 | 324, 432, 349, 453, 470, 422, 271, 356 |
| 3 | 369, 394, 367, 285, 303, 446, 474, 331 | 395, 354, 371, 372, 390, 352, 315, 307 |
| 4 | 445, 438, 357, 276, 350, 458, 526, 396 | 371, 369, 336, 369, 421, 364, 257, 316 |
| 5 | 304, 415, 371, 309, 275, 373, 416, 287 | 281, 313, 332, 320, 365, 332, 313, 277 |
| 6 | 322, 337, 345, 203, 283, 293, 390, 257 | 298, 381, 307, 349, 363, 416, 241, 246 |
| 7 | 356, 354, 387, 213, 273, 349, 472, 271 | 268, 322, 324, 323, 369, 327, 298, 294 |
| 8 | 193, 256, 296, 201, 227, 330, 325, 219 | 231, 297, 223, 199, 244, 209, 222, 226 |
| 9 | 405, 509, 512, 367, 393, 463, 579, 369 | 379, 467, 433, 401, 480, 469, 441, 411 |
| 10 | 379, 355, 327, 264, 281, 369, 408, 312 | 276, 323, 339, 313, 419, 417, 248, 291 |
| 11 | 180, 244, 208, 214, 219, 292, 247, 216 | 247, 214, 182, 233, 200, 173, 142, 157 |
| 12 | 350, 343, 355, 246, 318, 380, 436, 308 | 289, 298, 296, 293, 342, 291, 257, 256 |
| 13 | 257, 309, 273, 217, 180, 323, 332, 231 | 249, 289, 262, 281, 298, 269, 204, 221 |
| 14 | 313, 344, 330, 278, 326, 406, 443, 340 | 323, 267, 318, 324, 371, 340, 288, 292 |
| 15 | 250, 307, 298, 307, 221, 327, 366, 302 | 234, 178, 239, 291, 295, 206, 215, 224 |

---

## 2. Kernel 1：Naive 实现

### 2.1 核心思想

CUDA 编程模型中，计算被组织为三层层次结构：

```
Grid (网格)
├── Block (0,0) ── 包含最多 1024 个线程
├── Block (0,1)
├── ...
└── Block (gridDim.x-1, gridDim.y-1)
    └── Thread (threadIdx.x, threadIdx.y)
```

在 Naive Kernel 中，每个线程负责计算 C 的一个元素。对于 M=N=K=16，使用 `blockDim=(4,4), grid=(4,4)`，共 256 个线程，每个计算 C 矩阵的一个元素。

线程 (threadIdx.x, threadIdx.y) 在 block (blockIdx.x, blockIdx.y) 中计算：

```
i = blockIdx.x * 4 + threadIdx.x   // 全局行索引
j = blockIdx.y * 4 + threadIdx.y   // 全局列索引
C[i,j] = Σ(k=0..15) A[i,k] * B[k,j]
```

```
矩阵 C (16×16, GMEM)
└── Grid: 4×4 = 16 blocks (gridDim=(4,4), 16/4=4)
    └── Block: 16 threads (blockDim=(4,4))
        └── Thread: 1 element of C (内积 K=16)
            └── 每次 FMA: 从 GMEM 读 A[row,k] × B[k,col]
```

### 2.2 从第一性原理分析

**为什么 Naive 实现很慢？** 每个线程需要从全局内存 (Global Memory / GMEM) 加载 A 的一整行（16个float = 64B）和 B 的一整列（16个float = 64B）。总共 256 个线程 × 128B = 32KB 的 GMEM 读取量。但是相邻线程加载的 A 行不是连续的——这导致无法利用 GPU 的**合并内存访问（Coalescing）**特性，实际产生了大量浪费的 GMEM 带宽。在 A6000 上这个 Kernel 仅达到 300 GFLOPs（理论的 1.3%）。

### 2.3 具体计算演示

以 **C[0,0]** 为例，这是线程 (0,0) 在 Block (0,0) 中的计算结果：

```
C[0,0] = A[0,k] · B[k,0] 逐项求和，k = 0..15

k= 0:     5 ×     3 =     15
k= 1:     8 ×     7 =     56
k= 2:     9 ×     0 =      0
k= 3:     5 ×     6 =     30
k= 4:     0 ×     4 =      0
k= 5:     0 ×     8 =      0
k= 6:     1 ×     9 =      9
k= 7:     7 ×     2 =     14
k= 8:     6 ×     2 =     12
k= 9:     9 ×     1 =      9
k=10:     2 ×     4 =      8
k=11:     4 ×     6 =     24
k=12:     5 ×     9 =     45
k=13:     2 ×     2 =      4
k=14:     4 ×     0 =      0
k=15:     2 ×     6 =     12
─────────────────────────────────────────
总和: C[0,0] = 238 ✓ (匹配 Ground Truth)
```

以 **C[0,1]** 为例：

```
k= 0:     5 ×     8 =     40
k= 1:     8 ×     7 =     56
k= 2:     9 ×     4 =     36
k= 3:     5 ×     2 =     10
k= 4:     0 ×     9 =      0
k= 5:     0 ×     2 =      0
k= 6:     1 ×     8 =      8
k= 7:     7 ×     5 =     35
k= 8:     6 ×     0 =      0
k= 9:     9 ×     8 =     72
k=10:     2 ×     2 =      4
k=11:     4 ×     6 =     24
k=12:     5 ×     3 =     15
k=13:     2 ×     4 =      8
k=14:     4 ×     6 =     24
k=15:     2 ×     7 =     14
─────────────────────────────────────────
总和: C[0,1] = 346 ✓
```

以 **C[1,1]** 为例：

```
k= 0:     4 ×     8 =     32
k= 1:     7 ×     7 =     49
k= 2:     7 ×     4 =     28
k= 3:     9 ×     2 =     18
k= 4:     1 ×     9 =      9
k= 5:     7 ×     2 =     14
k= 6:     0 ×     8 =      0
k= 7:     6 ×     5 =     30
k= 8:     9 ×     0 =      0
k= 9:     9 ×     8 =     72
k=10:     7 ×     2 =     14
k=11:     6 ×     6 =     36
k=12:     9 ×     3 =     27
k=13:     1 ×     4 =      4
k=14:     0 ×     6 =      0
k=15:     1 ×     7 =      7
─────────────────────────────────────────
总和: C[1,1] = 340 ✓
```

以 **C[15,15]** 为例：

```
k= 0:     0 ×     8 =      0
k= 1:     0 ×     2 =      0
k= 2:     2 ×     7 =     14
k= 3:     9 ×     3 =     27
k= 4:     6 ×     4 =     24
k= 5:     0 ×     5 =      0
k= 6:     6 ×     4 =     24
k= 7:     7 ×     7 =     49
k= 8:     0 ×     5 =      0
k= 9:     3 ×     0 =      0
k=10:     9 ×     2 =     18
k=11:     0 ×     3 =      0
k=12:     3 ×     0 =      0
k=13:     4 ×     3 =     12
k=14:     7 ×     3 =     21
k=15:     5 ×     7 =     35
─────────────────────────────────────────
总和: C[15,15] = 224 ✓
```

### 2.4 内存访问模式的问题

```
Naive 的 A 矩阵访问模式（非合并，针对 全局内存 (GMEM)）:

  Thread (0,0): A[0,0] A[0,1] A[0,2] ... A[0,15]
  Thread (0,1): A[0,0] A[0,1] A[0,2] ... A[0,15]  ← 和 (0,0) 相同！
  Thread (1,0): A[1,0] A[1,1] A[1,2] ... A[1,15]
  ...

同一 warp 中的线程访问的是 A 的不同行（非连续 GMEM 地址），无法合并。
每个线程从 全局内存 (GMEM) 加载 2×16 个 float = 128 字节。
```

---

## 3. Kernel 2：全局内存合并访问

### 3.1 核心思想

**Warp** 是 GPU 执行的最小单元，包含 32 个线程。当同一个 warp 中的线程访问**连续的 全局内存 (GMEM) 地址**时，GPU 可以将这些访问合并为一次大事务（如 128B），大幅减少 GMEM 事务次数。

```
合并 GMEM 访问示例:
  Thread 0: addr 0x1000  ┐
  Thread 1: addr 0x1004  ├─→ 合并为一次 128B GMEM 读取
  Thread 2: addr 0x1008  │
  ...                    │
  Thread 31: addr 0x107C ┘
```

### 3.2 从第一性原理分析

**为什么合并访问有效？** GPU 的 全局内存 (GMEM) 控制器同一时间可以处理 32B、64B 或 128B 的事务。32 个线程各读取 4B 的 float，GMEM 地址连续时 = 128B，刚好一次事务完成。如果 GMEM 地址不连续，则需要 32 次独立事务 → GMEM 带宽利用率暴跌到 ~5%。

### 3.3 Kernel 2 的实现策略

将 `blockDim` 从 `(4,4)` 改为 `(16)`，即一维线程布局，但保持每 block 256 个线程（16×16 块的 256 个线程里实际上只用 16 个）。通过修改线程索引映射，使得连续 `threadIdx.x` 的线程访问连续的 全局内存 (GMEM) 地址：

```
// Kernel 2: 一维线程索引 + 合并 GMEM 访问
i = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)
j = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)
```

在我们的 demo 中：
- `blockDim = (16)`, `grid = (16, 16)` （共 256 blocks，每 block 16 线程）
- 但实际上一个 block 的 16 个线程分别负责 C 中同一行不同列的计算
- 连续 threadIdx.x 的线程访问 A 的**同一行**但不同元素 → GMEM 地址连续 → 合并访问

```
矩阵 C (16×16, GMEM)
└── Grid: 1×16 = 16 blocks (gridDim=(1,16), 16/1=16, 1D blockDim)
    └── Block: 16 threads (blockDim=(16))
        └── Thread: 1 element of C
            └── 访问模式: 同行线程连续访问 A, 合并为 128-bit 事务
```

### 3.4 效果

优化后 全局内存 (GMEM) 吞吐从 15GB/s 提升到 110GB/s，GFLOPs 从 300 提升到 ~2000（8.5% of cuBLAS）。在我们的 16×16 demo 中，计算结果与 Naive 完全一致，因为数学上等价。注意：Kernel 2 仍然不使用 共享内存 (SMEM)，所有数据都直接从 全局内存 (GMEM) 读取。

---

## 4. Kernel 3：共享内存缓存分块

### 4.1 核心思想

GPU 上除了大容量的全局内存 (Global Memory / GMEM, ~80GB)，还有片上共享内存 (Shared Memory / SMEM, ~48KB/block)。SMEM 的带宽可达 ~12TB/s，比 GMEM 的 ~768GB/s 高约 16 倍。

**策略**：将 A 和 B 的小块从 全局内存 (GMEM) 加载到 共享内存 (SMEM) 中，然后在 SMEM 上执行尽可能多的计算，减少 GMEM 访问。

### 4.2 从第一性原理分析

**为什么 SMEM 缓存有效？** 矩阵乘法有大量的数据复用——A 的每一行被 N 个输出元素复用，B 的每一列被 M 个输出元素复用。通过将数据从 GMEM 缓存到 SMEM，block 内的所有线程可以共享同一份数据，避免重复从 GMEM 加载。

**关键公式**：对每个 K-tile（大小 BK），block 内的线程协同将 `BM×BK` 的 A-tile 和 `BK×BN` 的 B-tile 从 全局内存 (GMEM) 加载到 共享内存 (SMEM)，然后所有线程在 SMEM 上执行内积。

```
矩阵 C (16×16, GMEM)
└── Grid: 4×4 = 16 blocks (gridDim=(4,4), 16/4=4)
    └── Block: BM=4, BN=4, BK=4 (blockDim=(4,4))
        ├── [GMEM→SMEM] As[4×4] + Bs[4×4] = 32 floats (128 bytes) per K-step
        ├── K 迭代: 16/4 = 4 steps
        └── Thread: 1 element, 从 SMEM 读取做 FMA
```

### 4.3 具体计算演示 (BM=BN=BK=4)

对于 Block (0,0)，第一轮 K-tile (bk_start=0)，从 全局内存 (GMEM) 加载到 共享内存 (SMEM) 的内容：

**As (4×4 SMEM tile of A, 行 0-3, 列 0-3):**

```
  [   5     8     9     5 ]
  [   4     7     7     9 ]
  [   8     8     3     9 ]
  [   0     3     9     2 ]
```

**Bs (4×4 SMEM tile of B, 行 0-3, 列 0-3):**

```
  [   3     8     8     0 ]
  [   7     7     7     2 ]
  [   0     4     7     0 ]
  [   6     2     9     5 ]
```

**线程 (0,0) 的第一轮部分内积（第一部分 dotIdx=0..3，在 SMEM 上计算）：**

```
Part1[0,0] = As[0,0]*Bs[0,0] + As[0,1]*Bs[1,0] + As[0,2]*Bs[2,0] + As[0,3]*Bs[3,0]

     5 ×    3 =     15
     8 ×    7 =     56
     9 ×    0 =      0
     5 ×    6 =     30
──────────────────────────────
Part1[0,0] = 101
```

经过 4 轮 K-tile (bk_start = 0, 4, 8, 12)，累加所有部分内积，最终得到：

```
Part1[0,0] = 101   (bk_start=0, 列 0-3,   加载到 SMEM)
Part2[0,0] =  23   (bk_start=4, 列 4-7,   从 GMEM 加载下一 tile 到 SMEM)
Part3[0,0] =  53   (bk_start=8, 列 8-11,  从 GMEM 加载下一 tile 到 SMEM)
Part4[0,0] =  61   (bk_start=12, 列 12-15, 从 GMEM 加载下一 tile 到 SMEM)
────────────────────────
C[0,0]     = 238 ✓
```

**验证 Part2[0,0] = 23:** (bk_start=4, As row 0, cols 4-7; Bs rows 4-7, col 0)
```
A[0,4]*B[4,0] + A[0,5]*B[5,0] + A[0,6]*B[6,0] + A[0,7]*B[7,0]
   0*4       +    0*8       +    1*9       +    7*2
 = 0 + 0 + 9 + 14 = 23 ✓
```

**验证 Part3[0,0] = 53:** (bk_start=8, As row 0, cols 8-11; Bs rows 8-11, col 0)
```
A[0,8]*B[8,0] + A[0,9]*B[9,0] + A[0,10]*B[10,0] + A[0,11]*B[11,0]
   6*2       +    9*1       +     2*4       +     4*6
 = 12 + 9 + 8 + 24 = 53 ✓
```

**验证 Part4[0,0] = 61:** (bk_start=12, As row 0, cols 12-15; Bs rows 12-15, col 0)
```
A[0,12]*B[12,0] + A[0,13]*B[13,0] + A[0,14]*B[14,0] + A[0,15]*B[15,0]
    5*9        +     2*2        +     4*0        +     2*6
 = 45 + 4 + 0 + 12 = 61 ✓
```

同样对于 C[0,1], C[0,2], C[0,3]（均在 Block (0,0) 内计算）：

```
C[0,1]: Part1=142 + Part2= 43 + Part3=100 + Part4= 61 = 346 ✓
C[0,2]: Part1=204 + Part2= 34 + Part3= 52 + Part4= 51 = 341 ✓
C[0,3]: Part1= 41 + Part2= 48 + Part3= 55 + Part4= 71 = 215 ✓
```

**不同 Block 的独立计算示例 — C[1,0]：**

```
Part1[1,0] = 115   (bk_start=0, SMEM tile 0)
Part2[1,0] =  72   (bk_start=4, SMEM tile 1)
Part3[1,0] =  91   (bk_start=8, SMEM tile 2)
Part4[1,0] =  89   (bk_start=12, SMEM tile 3)
────────────────────────
C[1,0]     = 367 ✓
```

**验证 Part1[1,0] = 115:** (bk_start=0, As row 1, Bs col 0)
```
   4*3 + 7*7 + 7*0 + 9*6 = 12 + 49 + 0 + 54 = 115 ✓
```

**验证 Part2[1,0] = 72:** (bk_start=4)
```
   1*4 + 7*8 + 0*9 + 6*2 = 4 + 56 + 0 + 12 = 72 ✓
```

**验证 Part3[1,0] = 91:** (bk_start=8)
```
   9*2 + 9*1 + 7*4 + 6*6 = 18 + 9 + 28 + 36 = 91 ✓
```

**验证 Part4[1,0] = 89:** (bk_start=12)
```
   9*9 + 1*2 + 0*0 + 1*6 = 81 + 2 + 0 + 6 = 89 ✓
```

**Tile 遍历的可视化:**

```
矩阵 A (16×16):              矩阵 B (16×16):
┌────────┬────────┬────────┬────────┐   ┌────────┬───┬───┬───┐
│ Tile 0 │ Tile 1 │ Tile 2 │ Tile 3 │   │ Tile 0 │   │   │   │
│ (0-3)  │ (4-7)  │ (8-11) │ (12-15)│   ├────────┼───┼───┼───┤
├────────┼────────┼────────┼────────┤   │ Tile 1 │   │   │   │
│   ..   │   ..   │   ..   │   ..   │   ├────────┼───┼───┼───┤
│        │        │        │        │   │ Tile 2 │   │   │   │
│        │        │        │        │   ├────────┼───┼───┼───┤
│        │        │        │        │   │ Tile 3 │   │   │   │
└────────┴────────┴────────┴────────┘   └────────┴───┴───┴───┘

每次迭代: 从 全局内存 (GMEM) 加载 A 的一个 4×4 Tile 和 B 的一个 4×4 Tile 
到 共享内存 (SMEM)，然后所有线程在 SMEM 上做内积，结果累加到
寄存器中的 partial sum
```

### 4.4 性能分析

这个 Kernel 达到 ~2980 GFLOPs（12.8% of cuBLAS），比上一版提升约 50%。提升没有预期的大，因为 L1 Cache 已经在上一版 Kernel 中提供了较好的命中率。`__syncthreads()` 的同步开销也限制了一部分性能。

**内存层次总结（Kernel 3）：**
- **全局内存 (GMEM) reads**: 加载 As, Bs tiles = 8192 bytes
- **全局内存 (GMEM) writes**: 写入最终 C = 1024 bytes
- **共享内存 (SMEM) writes**: 协同加载 As, Bs 到 SMEM = 8192 bytes
- **共享内存 (SMEM) reads**: 线程从 SMEM 读取做 FMA ≈ 32768 bytes

---

## 5. Kernel 4：1D Blocktiling

### 5.1 核心思想

Kernel 3 中每个线程仍只计算 C 的 **1 个元素**。这意味着每做一次 FMA（乘加），就需要从 共享内存 (SMEM) 加载 2 个 float。即算术强度（Arithmetic Intensity, AI）非常低。

**关键洞察**：让每个线程计算 C 的**多个元素**（一行中的多个），这样每个线程可以在寄存器中保存中间结果，对 SMEM 的加载可以跨多个输出复用。

```
Kernel 3 (每个线程 1 个结果):    Kernel 4 (每个线程 TM=4 个结果):
  Thread(0,0):                  Thread(0,0):
    Load As[row0], Bs[col0]       Load As[row0], Bs[col0]
    FMA → C[0,0]                  FMA → C[0,0]
    (下一个线程)                   Load As[row1], Bs[col0]
                                   FMA → C[1,0]
                                   ...
                                   (TM=4 个结果在一个线程中计算)
```

### 5.2 从第一性原理分析

**为什么每个线程计算多个结果更快？** 内核瓶颈在于 共享内存 (SMEM) 带宽。每个 FMA 需要 2 个 SMEM 读取。通过让每个线程做更多的计算（TM 个结果），可以复用 B 的值（`Btmp` 缓存技术），将每个结果的 SMEM 访问次数从 `2*BK` 降低到 `BK*(1 + 1/TM)`。当 TM=1 时为 2×，TM=8 时仅为 1.125×——减少了 44% 的 SMEM 访问。

> **Vulkan 联想备注：Cooperative Vector**
>
> 这里的 1D Blocktiling 可以帮助你联想到 Vulkan `VK_NV_cooperative_vector`，但不要把二者划等号。相似点是：一个执行单元逻辑上持有一组输出值，例如这里的 `threadResults[TM]`，它很像“一个小向量结果”。`cooperative vector` 也是让一个 shader invocation 逻辑上拥有自己的 vector，并在做 matrix-vector multiply 时由硬件背后协作加速。
>
> 差异是：本文这里仍是 GEMM 的一部分，一个 CUDA thread 明确计算 `C` 的一段；Vulkan cooperative vector 更偏向 `y = W x` 这种小型 MLP / matrix-vector 场景，协作细节由 Vulkan/SPIR-V 和驱动隐藏。阅读时可以把它当作“从一串输出值开始理解 matrix-vector 加速”的类比入口。

### 5.3 参数配置

| 参数 | 值 |
|------|-----|
| BM | 8 (block 覆盖 C 的行数) |
| BN | 8 (block 覆盖 C 的列数) |
| BK | 4 (K 维度分块大小) |
| TM | 4 (每个线程计算的行数) |

对于 M=N=K=16：
- 2×2 = 4 个 blocks
- 每个 block 有 `(BM/TM) * BN = (8/4) * 8 = 16` 个线程

```
矩阵 C (16×16, GMEM)
└── Grid: 2×2 = 4 blocks (gridDim=(2,2), 16/8=2)
    └── Block: BM=8, BN=8, BK=4 (blockDim=(8,2))
        ├── [GMEM→SMEM] As[8×4] + Bs[4×8] = 64 floats (256 bytes) per K-step
        ├── K 迭代: 16/4 = 4 steps
        └── Thread Tile: TM=4 (沿 M 方向 4 个 C 元素)
            └── Thread regs: 4×4 As 片段 + 4 Bs 标量
```

### 5.4 具体计算演示

**Block (0,0), Thread (threadRow=0, threadCol=0):**

该线程负责计算 C 的一个 1D 条带：`C[0,0], C[1,0], C[2,0], C[3,0]`

伪代码：
```
threadResults = [0, 0, 0, 0]  // TM=4, 保存在寄存器中

for bk_start in [0, 4, 8, 12]:      // 4 个 K-tile
    从 全局内存 (GMEM) 加载 As(8×4), Bs(4×8) 到 共享内存 (SMEM)
    
    for dotIdx in 0..3:               // BK=4
        Btmp = Bs[dotIdx, threadCol]  // 从 SMEM 缓存 B 的值到寄存器
        for resIdx in 0..3:           // TM=4
            threadResults[resIdx] += As[resRow+resIdx, dotIdx] * Btmp
            // As 和 Btmp 均从 SMEM 读取（As）或寄存器（Btmp）复用
```

预期结果（匹配 Ground Truth）：
```
C[0,0] = 238 ✓
C[1,0] = 367 ✓
C[2,0] = 402 ✓
C[3,0] = 369 ✓
```

**内存层次总结（Kernel 4）：**
- **全局内存 (GMEM) reads**: 4096 bytes（比 K3 减少 50%，因为 block 更大，block 数更少）
- **全局内存 (GMEM) writes**: 1024 bytes
- **共享内存 (SMEM) writes**: 4096 bytes（协同加载 As, Bs）
- **共享内存 (SMEM) reads**: ~8192 bytes（Btmp 缓存在寄存器中，减少了 SMEM 读取）

---

## 6. Kernel 5：2D Blocktiling — 提高算术强度

### 6.1 核心思想

Kernel 4 中每个线程计算 **1D 条带**（TM×1 个 C 元素）。进一步优化：让每个线程计算 **2D 小块**（TM×TN 个 C 元素），进一步提高数据复用率。

```
Kernel 4 (1D: TM×1):           Kernel 5 (2D: TM×TN):
  线程计算 C 的一列              线程计算 C 的一个子块
  
  ████                          ████
  ████  ← 一个线程              ████  ← 一个线程计算 4×4=16 个元素
  ████                          ████
  ████                          ████
```

### 6.2 从第一性原理分析

**为什么 2D 比 1D 更好？** 2D 计算模式允许同时复用 A 和 B 的数据：

- 1D (TM×1)：只能复用 B 的一个元素（缓存在寄存器）
- 2D (TM×TN)：可以同时复用 A 的一个元素（在 TN 个输出中）和 B 的一个元素（在 TM 个输出中）

这进一步减少了每个结果的平均 共享内存 (SMEM) 访问次数。

### 6.3 参数配置 (demo)

| 参数 | 值 |
|------|-----|
| BM | 8 |
| BN | 8 |
| BK | 4 |
| TM | 4 |
| TN | 4 |

每个线程计算 TM×TN = 4×4 = 16 个 C 元素。

对于 M=16, N=16：
- 2×2 = 4 个 blocks
- 每个 block 有 `(BM/TM) * (BN/TN) = (8/4)*(8/4) = 4` 个线程

```
矩阵 C (16×16, GMEM)
└── Grid: 2×2 = 4 blocks (gridDim=(2,2), 16/8=2)
    └── Block: BM=8, BN=8, BK=4 (blockDim=(2,2), 4 threads/block)
        ├── [GMEM→SMEM] As[8×4] + Bs[4×8] = 64 floats per K-step
        ├── K 迭代: 16/4 = 4 steps
        └── Thread Tile: TM=4, TN=4 (线程计算 4×4=16 个 C 元素)
            └── Thread regs: As[4×4] + Bs[4×4], FMA 全部在寄存器
```

### 6.4 具体计算演示

**Block (0,0), Thread (thr_row=0, thr_col=0):**

该线程负责计算 C 的 **4×4 子块**：`C[0:4, 0:4]`

```
线程的子块（所有值在寄存器中累积）:
  ┌─────────────────────────────────────────┐
  │ C[0,0]    C[0,1]    C[0,2]    C[0,3]   │
  │   238       346       341       215     │
  │                                         │
  │ C[1,0]    C[1,1]    C[1,2]    C[1,3]   │
  │   367       340       378       243     │
  │                                         │
  │ C[2,0]    C[2,1]    C[2,2]    C[2,3]   │
  │   402       412       456       331     │
  │                                         │
  │ C[3,0]    C[3,1]    C[3,2]    C[3,3]   │
  │   369       394       367       285     │
  └─────────────────────────────────────────┘
```

伪代码：
```
threadResults = 4×4 零矩阵（在寄存器中）

for bk_start in [0, 4, 8, 12]:
    从 全局内存 (GMEM) 加载 As(8×4), Bs(4×8) 到 共享内存 (SMEM)
    
    for dotIdx in 0..3:                // BK=4
        for resRow in 0..3:            // TM=4
            for resCol in 0..3:        // TN=4
                threadResults[resRow,resCol] += As[resRow, dotIdx] * Bs[dotIdx, resCol]
                // As 和 Bs 从 共享内存 (SMEM) 读取
```

所有 16 个元素均与 Ground Truth 一致 ✓

**内存层次总结（Kernel 5）：**
- **全局内存 (GMEM) reads**: 4096 bytes（与 K4 相同）
- **全局内存 (GMEM) writes**: 1024 bytes
- **共享内存 (SMEM) writes**: 4096 bytes
- **共享内存 (SMEM) reads**: ~32768 bytes（2D 计算模式增加 SMEM 复用）

---

## 7. Kernel 6：向量化内存访问

### 7.1 核心思想

原文的 Kernel 6 其实包含两层向量化，不能混在一起理解：

1. **先转置 `As`，让共享内存读取也能向量化**：Kernel 5 中 `Bs` 的 shared memory load 已经容易被编译器合并成 `LDS.128`，但 `As` 的访问方向不够连续。Kernel 6 把 `As` 在写入共享内存时转置，使后续从 SMEM 读 `As` 片段时也能形成连续的 128-bit 读取。
2. **再用 `float4` 向量化 GMEM load/store**：从全局内存搬到共享内存时，使用 `float4` 一次搬 4 个连续 FP32，减少全局内存加载/存储指令数量。

所以这一步不是简单地说“SMEM 支持 float4”，而是：**通过内存布局和对齐承诺，让 SMEM 读取与 GMEM 搬运都尽量变成 128-bit 指令。**

```
非向量化 (SMEM):                    向量化 (SMEM float4):
  ld.shared.f32 r1, [addr]           ld.shared.v4.f32 {r1,r2,r3,r4}, [addr]
  ld.shared.f32 r2, [addr+4]
  ld.shared.f32 r3, [addr+8]
  ld.shared.f32 r4, [addr+12]
```

### 7.2 从第一性原理分析

**为什么向量化有效？** 它首先减少的是指令数量和内存事务组织成本：4 个连续 float 原本可能需要 4 条 32-bit 指令，现在可以用一条 128-bit 指令表达。对 SMEM 来说，转置 `As` 是为了让读取方向连续，方便编译器生成 `LDS.128`；对 GMEM 来说，`float4` 是为了让全局内存搬运更宽。

原文还特别解释了 `reinterpret_cast<float4*>` 的意义：编译器不能凭空证明传入 kernel 的 `float*` 指针一定 128-bit 对齐，所以它不会随便把普通 `float` load 改成 `LDG.E.128`。`reinterpret_cast<float4*>` 相当于程序员向编译器承诺“这里是对齐的，可以按 128-bit 读”。相比之下，shared memory 是 kernel 内部管理的内存，编译器更容易判断布局并自动生成向量化 SMEM load。

在我们的 demo 中（BK=4），As 的每一行恰好是 4 个连续的 float，很适合用来展示 `float4` 搬运；真实原文里的重点则是“为了让这些连续性成立，必要时要调整 shared memory 中的布局”。

```
(Same hierarchy as K5)
矩阵 C (16×16, GMEM)
└── Grid: 2×2 = 4 blocks (gridDim=(2,2), 16/8=2)
    └── Block: BM=8, BN=8, BK=4 (blockDim=(2,2))
        ├── [GMEM→SMEM] float4 向量加载: As[8×4] 用 8 次 128-bit 事务
        ├── [SMEM布局]  转置 As，让后续 SMEM→RF 读取方向连续
        ├── [SMEM→RF]  尽量生成 128-bit LDS 指令，减少 shared load 指令数
        └── Thread Tile: TM=4, TN=4 → 16 elements/thread
```

### 7.3 在 16×16 demo 中的体现

对于 BK=4，As 的每行恰好是 4 个 float，一次 `float4` 从 GMEM 加载到 SMEM：
```
As[0] = {5, 8, 9, 5}  ← 一次 float4 加载 (GMEM → SMEM)
```
这与 Kernel 5 计算结果完全一致。原始文章中，`As` 转置带来的 SMEM 向量化只有小幅收益；后续 `float4` GMEM load/store 才是更明显的收益来源，整体将性能从 68.7% 提升到 78.4% of cuBLAS。

---

## 8. Kernel 10：Warp 级分块

### 8.1 核心思想

一个 Block 内的线程进一步按 Warp 组织，在 block tiling 和 thread tiling 之间增加一层 **warp tiling**。原文强调：warp 不是 CUDA C 代码里显式出现的普通对象，而是硬件调度单位；每个 warp 有 32 个线程，会被 warp scheduler 作为一组来发射和调度。

这一层的目的不是引入某个单一指令技巧，而是把并行层级说清楚并组织得更适合硬件：

- block 级别：不同 block 分配到不同 SM。
- warp 级别：同一 block 内的不同 warp 可以在不同 warp scheduler 上并行推进，也可以在同一 scheduler 上交错推进。
- thread tile 级别：每个线程维护自己的寄存器 accumulator，提供有限但重要的 ILP。
- 内存访问级别：SMEM bank conflict 发生在同一 warp 内，warp tiling 可以让共享内存访问和寄存器复用更规则。

**策略**：
1. 将 Block 的工作按 Warp Tile 划分
2. 每个 Warp 负责 C 的一个子区域
3. Warp 内线程从共享内存加载自己的 register fragments
4. 通过更规则的 warp/thread 层级，提高调度效率、降低 bank conflict 风险，并改善寄存器缓存局部性

### 8.2 Demo 配置

| 参数 | 值 | 含义 |
|------|-----|------|
| BM | 8 | Block 覆盖的行 |
| BN | 8 | Block 覆盖的列 |
| BK | 8 | K-tile 大小 |
| WM | 4 | Warp 协作的行数 |
| WN | 4 | Warp 协作的列数 |
| WMIter | 2 | Warp 内沿 M 的迭代次数 |
| WNIter | 2 | Warp 内沿 N 的迭代次数 |
| TM | 4 | 每线程计算的行数 |
| TN | 4 | 每线程计算的列数 |

Warp 区域 = `WMIter * TM × WNIter * TN = 8 × 8` = 整个 Block

```
矩阵 C (16×16, GMEM)
└── Grid: 2×2 = 4 blocks (gridDim=(2,2), 16/8=2)
    └── Block: BM=8, BN=8, BK=4 (blockDim 取决于 warp 展开方式)
        ├── 每 block: WM_ITER=2 × WN_ITER=2 = 4 个 warp-tile
        ├── [GMEM→SMEM] As[8×4] + Bs[4×8] = 64 floats (256 bytes) per K-step
        ├── K 迭代: 16/4 = 4 steps, double-buffered
        ├── Warp Tile: WM=4, WN=4 (demo 中用逻辑 warp-tile 展示 4×4 子区域)
        │   └── Thread Tile: TM=4, TN=4 (每线程 4×4=16 元素)
        └── 说明: 真实 CUDA warp 固定为 32 threads；demo 为了可视化，只画出 4 个逻辑计算 tile
```

### 8.3 Warp 工作划分 (Block 0,0)

```
Block (0,0) 覆盖 C[0:8, 0:8]:
┌───────────────────────┬───────────────────────┐
│ Thread (0,0)          │ Thread (0,1)          │
│ C[0:4, 0:4]           │ C[0:4, 4:8]           │
│                       │                       │
│  238   346   341   215 │  316   377   322   282 │
│  367   340   378   243 │  352   476   433   357 │
│  402   412   456   331 │  349   468   559   385 │
│  369   394   367   285 │  303   446   474   331 │
├───────────────────────┼───────────────────────┤
│ Thread (1,0)          │ Thread (1,1)          │
│ C[4:8, 0:4]           │ C[4:8, 4:8]           │
│                       │                       │
│  445   438   357   276 │  350   458   526   396 │
│  304   415   371   309 │  275   373   416   287 │
│  322   337   345   203 │  283   293   390   257 │
│  356   354   387   213 │  273   349   472   271 │
└───────────────────────┴───────────────────────┘
```

这里画的是 4 个"逻辑计算 tile"，用于解释 warp tile 如何覆盖 block 的 C 区域。真实 CUDA kernel 中 warp 仍然是 32 个线程，线程之间的具体分工比 demo 图更细；demo 不应该被理解成“一个 warp 只有 4 个线程”。

### 8.4 从第一性原理分析

**为什么 Warp tiling 有效？** 原文的重点是增加一个与硬件调度一致的层级。没有 warptiling 时，我们只有 block tile 和 thread tile 两层；加入 warptiling 后，block 内的工作被分配给多个 warp tile，warp scheduler、SMEM bank conflict 范围、寄存器缓存局部性都能被更明确地利用。

在这一路径里，数据仍然会经历 GMEM → SMEM → RF，warp 内线程从共享内存加载各自的 register fragments，再执行寄存器里的 FMA。更准确的说法是：warptiling 让每个 warp 负责的输出区域、SMEM 读取模式和寄存器复用模式更清晰，最终更接近 cuBLAS 的手写层级组织。原始文章达到 21.8 TFLOPs（93.7% of cuBLAS）。

> **Vulkan 联想备注：Cooperative Matrix**
>
> 这里最适合联想到 Vulkan `VK_KHR_cooperative_matrix`。相似点是：矩阵小块不再被理解成“某个线程自己的普通数组”，而是由一组执行单元协作完成。本文用 CUDA 的 warp tile 来显式组织 warp/thread 的层级；Vulkan cooperative matrix 则把一个中等大小的 matrix 类型定义成“存储和计算分布在某个 scope 的多个 invocation 上”，通常是 subgroup。
>
> 关键联想：本文的 Warp Tile 是手写展开的协作策略；Vulkan cooperative matrix 是 API/IR 层暴露的协作矩阵原语。你读到 `Warp Tile`、`warp 内线程协作`、`register fragment` 时，就可以想到 cooperative matrix 背后的目标也是把小矩阵乘法交给一组 invocation 和专用矩阵硬件去完成。

**内存层次总结（Kernel 10）：**
- **全局内存 (GMEM) reads**: 4096 bytes
- **全局内存 (GMEM) writes**: 1024 bytes
- **共享内存 (SMEM) writes**: 4096 bytes（协同加载）
- **共享内存 (SMEM) reads**: 仍然存在，但访问模式更规则
- **关键提升**: 增加 warp 这一层并行和数据复用组织，让调度、SMEM 访问和寄存器 accumulator 更贴近硬件执行方式

---

## 9. 性能分析：Occupancy、算术强度与 Roofline 模型

### 9.1 Occupancy

**定义**：每个 SM 上的活跃 warp 数 / 最大可能活跃 warp 数。

对 Kernel 3（BM=BN=BK=32 在 A6000 上）:
```
每个 Block: 1024 threads = 32 warps, 37 registers/thread, 8KB 共享内存 (SMEM)
每个 SM:    max 1536 threads = 48 warps, 100KB 共享内存 (SMEM)

共享内存 (SMEM) 限制:  100KB / 8KB = 12.5 → 最多 12 blocks
Thread 限制: 1536 / 1024 = 1.5 → 最多 1 block  ← 瓶颈！
Register 限制: 65536 / (1280*32) ≈ 1.6 → 最多 1 block

Occupancy = 32 active warps / 48 max warps = 66.7%
```

**关键洞察**：高 Occupancy 不总是好事。当算术强度很高时，低 Occupancy 足以隐藏延迟（因为有足够多的独立指令可发射）。这在 Volkov 的博士论文中被总结为"Cusp Behavior"。

### 9.2 算术强度 (Arithmetic Intensity)

**定义**：AI = FLOPs / Bytes transferred

```
总 FLOPs = 2 × M × N × K = 2 × 16³ = 8,192

各 Kernel 的内存传输 (demo 16×16):

K1 Naive:    每个线程从 全局内存 (GMEM) 加载 2×16 floats × 256 threads
             ≈ 32KB GMEM reads (最坏情况, 无缓存)
             共享内存 (SMEM): 未使用
             AI = 8192 / 32768 ≈ 0.24 FLOPs/Byte (vs GMEM)

K3 SMEM:     BM×BK + BK×BN = 4×4+4×4 = 32 floats/K-tile/block
             从 全局内存 (GMEM) 加载: × 4 K-tiles × 16 blocks × 4B = 8,192B
             共享内存 (SMEM) writes: 8,192B (协同加载 As, Bs)
             共享内存 (SMEM) reads: ~32,768B (线程 FMA 读取)
             AI (vs GMEM) ≈ 1.0 FLOPs/Byte

K4 1D:       与 K3 相比, GMEM reads 减半 (block 更大, block 数更少 = 4)
             全局内存 (GMEM) reads: ~4,096B
             共享内存 (SMEM) reads: ~8,192B (Btmp 缓存在寄存器)
             AI (vs GMEM) ≈ 2.0 FLOPs/Byte

K5 2D:       全局内存 (GMEM): 同 K4 (~4,096B)
             共享内存 (SMEM) reads: ~32,768B
             AI (vs GMEM) ≈ 2.0 FLOPs/Byte
             (2D 复用提高 SMEM 效率, GMEM 用量不变)

K10 Warp:    全局内存 (GMEM) reads: ~4,096B
             BK=8 意味着 2 个 K-tile, 每个 tile 做更多计算
             AI (vs GMEM) ≈ 2.0 FLOPs/Byte (demo scale)
             注意：demo 的 2.0 是按 16×16 小矩阵和简化 tile 口径计算；
             原文大规模 kernel 会因为更大的 tile、warp/register 复用和更少的边界开销获得更高有效 AI。
```

### 9.3 内存流量汇总表 (GMEM vs SMEM)

| Kernel | 全局内存 (GMEM) Reads | 全局内存 (GMEM) Writes | 共享内存 (SMEM) Reads | 共享内存 (SMEM) Writes | AI (vs GMEM) |
|--------|----------------------|----------------------|---------------------|----------------------|-------------|
| K1 Naive | 32,768 B | 1,024 B | 0 B | 0 B | 0.24 |
| K2 Coalescing | 32,768 B | 1,024 B | 0 B | 0 B | 0.24 |
| K3 SMEM | 8,192 B | 1,024 B | ~32,768 B | 8,192 B | 1.00 |
| K4 1D Blocktiling | 4,096 B | 1,024 B | ~8,192 B | 4,096 B | 2.00 |
| K5 2D Blocktiling | 4,096 B | 1,024 B | ~32,768 B | 4,096 B | 2.00 |
| K10 Warptiling | 4,096 B | 1,024 B | ~32,768 B | 4,096 B | 2.00 |

> 注：demo 规模 (16×16) 下 GMEM 用量相近。实际大规模矩阵 (4096×4096) 下，SMEM 缓存的收益呈数量级放大：K3 的 GMEM 访问相比 K1 可减少 16×，K10 通过 warp 级寄存器复用进一步将 SMEM 压力减半。

### 9.4 Roofline 模型

```
峰值 fp32 算力 (A6000):          ~30 TFLOPs/s
峰值 全局内存 (GMEM) 带宽 (A6000):  ~768 GB/s
峰值 共享内存 (SMEM) 带宽 (A6000):  ~12 TB/s (≈16× GMEM)
Ridge Point (vs GMEM):           30,000 / 768 ≈ 39 FLOPs/Byte

┌──────────────────────────────────────────────┐
│                     Roofline                 │
│  TFLOPs/s                                    │
│  30 ┤                              ╭─────────│ ← 算力上限 (Compute Bound)
│     │                           ╭──╯         │
│  20 ┤                     ╭─────╯  K10       │
│     │              ╭──────╯   K6             │
│  10 ┤       ╭──────╯  K5                    │
│     │  ╭────╯ K4                             │
│   1 ┤──╯ K3                                  │
│     ├────┼────┼────┼────┼────┼────┼──────    │
│     0.1   1   10  39  100              AI    │
│     ← Memory Bound →│← Compute Bound →       │
└──────────────────────────────────────────────┘

Demo 16×16 口径：
K1/K2:    AI ≈ 0.24, 深度 Memory Bound (全局内存 (GMEM) 带宽瓶颈)
K3:       AI ≈ 1.0,  Memory Bound，但 GMEM 读取已经明显下降
K4-K6:    AI ≈ 2.0,  仍是 Memory Bound，主要收益来自更好的复用和指令效率
K10:      AI ≈ 2.0,  demo 规模下 GMEM 数据量与 K4-K6 相近

原文大规模 kernel 口径：
K10 通过更大的 tile、warp 级寄存器复用、向量化和更好的调度，把有效 AI 和实际吞吐继续推高，接近 cuBLAS 的性能区间。
cuBLAS:   AI ≈ 245,  深度 Compute Bound (利用了 Tensor Cores)
```

**关键结论**：所有优化都在做同一件事——用更少的字节传输完成相同的 FLOPs：
1. 数据从 全局内存 (GMEM) 搬到 共享内存 (SMEM) → 利用 ~16× 带宽优势
2. 数据从 共享内存 (SMEM) 复用到寄存器 → 进一步减少 SMEM 访问
3. 每个线程做更多计算（提高 TM, TN）→ 提高每字节传输的计算量
4. 当 AI 超过 Ridge Point（39 FLOPs/Byte），kernel 变成 Compute Bound

---

## 10. 总结

| Kernel | 优化技术 | 关键原理 | 内存层次收益 | 相对 cuBLAS |
|--------|---------|---------|------------|------------|
| 1 Naive | 无 | 基础矩阵乘法，每个线程从 全局内存 (GMEM) 独立加载 | GMEM: 32KB reads | 1.3% |
| 2 Coalescing | 合并 GMEM 访问 | 连续 GMEM 地址 → 减少内存事务 | GMEM: 合并但同数据量 | 8.5% |
| 3 SMEM | 共享内存 (SMEM) 缓存 | 片上 SMEM 缓存减少 GMEM 访问 | GMEM↓4×, SMEM 启用 | 12.8% |
| 4 1D Blocktiling | 每线程计算 TM=4 个输出 | 寄存器缓存 B 值，减少 SMEM 读取 | SMEM reads↓4× | 36.5% |
| 5 2D Blocktiling | 每线程计算 TM×TN=16 个输出 | 同时复用 A 和 B，最大化寄存器和 SMEM 效率 | SMEM 复用最大化 | 68.7% |
| 6 Vectorized | As 转置 + float4 GMEM 搬运 | 让 SMEM 读取和 GMEM 搬运都尽量变成 128-bit 指令 | 指令数和事务组织成本↓ | 78.4% |
| 10 Warptiling | Warp 级分块 + 寄存器复用 | 在 block tile 与 thread tile 之间加入硬件调度层级 | 调度、SMEM 访问、寄存器局部性更规则 | 93.7% |

**第一性原理总结**：

1. **内存层次结构是关键**：全局内存 (GMEM, 慢, ~768GB/s) → L1/L2 Cache → 共享内存 (SMEM, 快, ~12TB/s) → Register (最快)。每次将数据"推"到更靠近计算单元的地方，性能就提升一个台阶。

2. **算术强度是统一的优化目标**：所有技术本质上都在提高 AI = FLOPs / Bytes。Naive Kernel 每个 FMA 需要从 全局内存 (GMEM) 加载 ~128B（AI≈0.008），而 cuBLAS 的 AI ≈ 245 FLOPs/Byte（利用 Tensor Cores）。

3. **GPU 是吞吐量架构，不是延迟架构**：单个指令延迟（FMA=4 cycles）不是瓶颈，关键是能否持续"喂饱"计算单元。高 Occupancy 帮助隐藏延迟，但高 AI 时不需要高 Occupancy。

4. **渐进式优化是理解系统的唯一方法**：每一步优化（30%-100% 提升）都揭示了一个硬件特性：
   - 全局内存 (GMEM) 合并 → 内存控制器批量事务
   - 共享内存 (SMEM) → 片上存储 + 数据复用
   - 寄存器复用 → 每线程多输出减少 SMEM 压力
   - Warp 级分块 → 显式利用 warp 调度层级，让 SMEM 访问和寄存器复用更规则

> **验证说明**：本文所有数值计算均可通过配套 Python 脚本 `demo-01-siboehm-cuda-matmul-worklog.py` 复现。脚本使用 `numpy.random.seed(1)` 和 `np.random.randint(0, 10, (16,16)).astype(np.float32)` 生成 0-9 整数矩阵，并逐步验证每个 Kernel Stage 的输出与 numpy `@` 运算符的结果一致（最大误差 = 0，全部 PASS）。

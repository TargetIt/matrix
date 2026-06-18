Source 02 · 从代码执行过程讲清楚
# NVIDIA_SGEMM_PRACTICE：七个 kernel，每一步代码到底在搬什么、算什么

> 🚀 **【第一性原理导读：为什么一份代码能让性能翻6倍？】**
> 
> **第一步：明确极限在哪里（Roofline 模型直觉）**
> 任何一段程序在电脑里跑，都会受到两个屋顶（极限）的压制：一个是**计算天花板**（CPU/GPU一秒能算多少次），另一个是**内存带宽天花板**（一秒能搬多少数据）。矩阵乘法天生就是一个“非常渴求数据”的算法。
>
> **第二步：看透复用的本质（Data Reuse）**
> 矩阵乘法的**第一性原理**告诉我们：计算 $C_{0,0}$ 需要用到 $A$ 的第一行，计算 $C_{0,1}$ 同样需要用到 $A$ 的第一行。**同样的数据，在矩阵乘法中是被无数次重复需要的**。
>
> **第三步：层层递进的性能压榨路线**
> 这篇文章展现的，就是一部“极限压榨显卡”的实战记录。每一代 Kernel 都在解答一个更深层次的问题：
> - **Kernel 1（原始）**：没有复用。需要一次，就去几公里外的大仓库（全局内存）拿一次。
> - **Kernel 2（共享内存）**：**群体复用**。大家凑钱建一个近处的小中转站（Shared Memory），去大仓库拿一车回来，大家共享。
> - **Kernel 3/4（Thread Tile）**：**个体复用**。发现中转站拿货还是有点慢，于是让工人拿到货后，同时给手边的 8 个甚至 64 个箱子打包，彻底榨干这一次拿货的价值。
> - **Kernel 5（寄存器缓存）**：**极致复用**。把接下来几秒钟要用的货，直接死死攥在自己手里（寄存器 Register），连中转站都不去了。
> - **Kernel 7（数据预取/流水线）**：**时间复用**。不仅要复用数据，还要复用时间！左手干活的同时，右手去拿下一批材料（Double Buffering）。
>
> 这就是从“零复用”走向“全方位立体复用”的全过程。理解了这个本质，你就能看懂那些复杂的代码。


这份仓库最适合拿来看代码路径。它不是只讲概念，而是把优化拆成多个 CUDA kernel。下面用“一个小矩阵、一段伪代码、一组生活类比”把每个 kernel 的意图讲清楚。

原始仓库：wangzyon/NVIDIA_SGEMM_PRACTICE

## 原文线索

> 📌 **原文主题：** Step-by-step optimization of CUDA SGEMM。仓库用多个 CUDA kernel 展示朴素实现、共享内存、thread tile、寄存器缓存、FLOAT4、双缓存等优化。

> 📌 **关键数据：** 本站可视化采用 README 中 RTX 3090、M=N=K=5120 的性能阶梯：从约 2.26 TFLOP/s 提升到约 13.63 TFLOP/s。

> 📌 **原文结构：** 每个 kernel 对应一个代码版本，性能表负责说明收益，代码负责说明线程如何映射、数据如何搬运。

> 💡 **小白通俗解析：** 读这个仓库，最有效的方法不是盯着最终最快版本，而是比较相邻两个版本：这一步新增了什么变量？多了哪块共享内存？一个线程负责的输出是不是变多了？这些变化就是性能提升的来源。

## 先用一个数字例子，把“线程负责 C 元素”讲透

先看一个 2×2 的具体乘法。假设：

```cpp
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

C[0,1] = A[0,0]×B[0,1] + A[0,1]×B[1,1]
       = 1×6 + 2×8
       = 22
```

> 💡 **小白通俗解析：** 如果一个线程负责 C[0,1]，它就要读 A 的第 0 行 `[1,2]`，再读 B 的第 1 列 `[6,8]`，然后做两次乘加。真实 SGEMM 只是把这个小例子放大到成千上万个线程、成千上万个元素。

假设 C 是 4×4。朴素版本里，一个线程负责一个 C 元素。threadIdx.x 可以映射成二维坐标：

```cpp
tx = threadIdx.x % BN
ty = threadIdx.x / BN
```

如果 BN=4，那么：

| threadIdx.x | ty | tx | 负责输出 |
|---|---|---|---|
| 0 | 0 | 0 | C[0,0] |
| 1 | 0 | 1 | C[0,1] |
| 4 | 1 | 0 | C[1,0] |
| 15 | 3 | 3 | C[3,3] |

> 💡 **小白通俗解析：** 这就是代码里的第一条主线：线程编号不是随便用的，它会变成矩阵里的行列坐标。后面所有 BM、BN、TM、TN 参数，本质都在改变“一个线程负责多少输出，以及这些输出在 C tile 的什么位置”。

## 先建立一个总原则：FLOPs 不变，数据复用变了

很多 SGEMM 优化容易让人误解，好像引入某个技术后“计算量变少了”。严格说，矩阵乘法的数学计算量没有变。假设 C 是 M×N，K 是内积长度，那么总 FMA 数永远是：

总 FMA = M × N × K

优化真正改变的是：为了完成这些 FMA，要从 GMEM、SMEM、寄存器里读取多少次数据；每读一次 A 或 B，能服务多少次乘加。

> 💡 **小白通俗解析：** 所以后面看每一步技术，请用同一个问题检查它：引入之前，每读一次数据能做几次 FMA？引入之后，每读一次数据能做几次 FMA？如果答案变大，说明数据复用变好了。

| 技术 | 引入前 | 引入后 | 主要好处 | 主要代价 |
|---|---|---|---|---|
| Shared Memory | 线程反复从 GMEM 读 A/B | block 合作搬 A/B tile 到 SMEM，再复用 | 减少慢速 GMEM 重复读取 | 需要同步；占用 SMEM |
| 1D Thread Tile | 一个线程算 1 个 C | 一个线程沿一个方向算多个 C | 一个 B 或 A 值服务多个 accumulator | 寄存器数量增加 |
| 2D Thread Tile | 线程只做单点或一条线 | 线程算 TM×TN 的 C 小块 | A/B fragment 形成 outer product，复用更强 | 寄存器压力更高，调参更敏感 |
| Register Cache | 内层循环频繁读 SMEM | 当前 A/B fragment 先放进寄存器 | 减少 SMEM 访问，FMA 喂数更快 | 寄存器占用上升 |
| FLOAT4 | 一次 load 一个 float | 一次 load 4 个连续 float | 减少访存指令和内存事务 | 要求连续、对齐、后续能消费 |
| Double Buffer | 先搬 tile，再计算，搬运和计算串行 | 算当前 tile 时预取下一 tile | 隐藏部分 GMEM 延迟 | 代码复杂；SMEM 使用翻倍 |

## Kernel 1：朴素实现，最容易懂，也最容易慢

一个线程负责一个 C 元素。例如 thread 1 负责 C[0,1]。它会读取 A 的第 0 行和 B 的第 1 列：

```cpp
tmp = 0
for k = 0..K-1:
    tmp += A[0,k] * B[k,1]
C[0,1] = tmp
```

> 💡 **小白通俗解析：** 这像一个学生负责算表格里的一个格子。为了算这个格子，他把横向的一整排资料和竖向的一整列资料拿出来，对应相乘再相加。方法完全正确，但每个学生都自己去拿资料，重复很多。

> 💡 **小白通俗解析：** 可视化里浅红色的一行 A 和一列 B，表示这个线程完整 K-loop 会扫过的范围；深红色表示当前这一小步正在读取的 A[row,k] 和 B[k,col]。

## Kernel 2：Shared Memory，大家合伙搬货

Kernel 2 开始让一个 block 的线程协作。每个线程从 GMEM 搬一点 A/B tile 到 Shared Memory，然后所有线程从 SMEM 里读。

```cpp
As[ty, tx] = A[global_row, global_k]
Bs[ty, tx] = B[global_k, global_col]
__syncthreads()

for i in 0..BK-1:
    tmp += As[ty, i] * Bs[i, tx]
```

> 💡 **小白通俗解析：** 这一步解决的是“重复从 GMEM 取同样数据”的问题。A/B tile 先被放到 SMEM，block 内线程共享使用。每个线程仍然可以只负责一个 C，但它读 A/B 的地点变近了。

> 💡 **小白通俗解析：** 以前每个人都跑仓库；现在每个人分工拿一点货到桌上，大家等桌子摆满，再一起从桌上拿。`__syncthreads()` 就是“等大家都摆完再开工”。

## Kernel 3：一维 Thread Tile，一个线程算多个 C

Kernel 3 让一个线程沿 M 方向负责多个输出，比如 TM=4、TN=1，thread 可能负责：

```cpp
C[0, col]
C[1, col]
C[2, col]
C[3, col]
```

这样同一个 B 值可以复用 4 次：

```cpp
b = Bs[k, col]
acc0 += As[row+0, k] * b
acc1 += As[row+1, k] * b
acc2 += As[row+2, k] * b
acc3 += As[row+3, k] * b
```

> 💡 **小白通俗解析：** 关键变化是：线程不再只维护一个 accumulator，而是维护多个 accumulator。一个从 B 读来的值，能同时服务多个 C 输出。数据搬一次，算得更多。

> 💡 **小白通俗解析：** 像厨师拿到一勺酱，不只给一碗面加，而是给四碗面都加。酱只拿一次，产出变多。

## 彻底搞懂 2D Thread Tile：为什么数据变少了？

很多初学者卡在这一步：**“一个线程到底在维护什么？为什么 K 循环 4 次（0~3）？为什么 A 被读取了 16 次？”**

别急，我们用一个最具体的例子来拆解。假设我们要计算结果矩阵 $C$ 中的一个小方块，大小是 **4×4（即 16 个数字）**。假设内积长度 **$K=4$**。

在数学上，要算出这 16 个数字，总共需要的乘加计算（FMA）次数是固定的：
**$4 × 4 × 4 = 64$ 次 FMA。**

### 第一种情况：没有 Thread Tile（1个工人只管 1 个箱子）

假设我们雇佣了 **16 个工人（16 个线程）**，每人只负责算出 $C$ 里面的 **1** 个数字。

- 工人 1 负责算 $C_{0,0}$。他需要去拿 $A$ 的第 0 行、$B$ 的第 0 列。因为 $K=4$，他要分 **4 轮（k=0, 1, 2, 3）** 去拿，每次拿 1 个 $A$ 和 1 个 $B$ 乘起来。
- 工人 2 负责算 $C_{0,1}$。他也需要去拿 $A$ 的第 0 行、$B$ 的第 1 列... 
- **⚠️ 发现问题了吗？** 工人 1 和工人 2，他们都需要用到 **$A$ 的第 0 行**！但因为他们是两个人，他们各自跑去大仓库拿了一遍。

**结果就是：**
为了做完这 64 次计算，16 个工人总共去拿了 **64 次 $A$ 和 64 次 $B$**（总计 128 次拿货）。

### 第二种情况：2D Thread Tile（1个超级工人管 16 个箱子）

现在我们开掉 15 个人，只留下 **1 个超级工人（1个线程）**。我们对他说：“这 4×4=16 个箱子，全归你一个人管！”

**1. 这个工人到底在维护什么？**
他在自己的办公桌上摆了 16 个空盒子。在代码里，这就是 **`acc[4][4]` 这个寄存器数组**。这就是他要维护的东西。

**2. 为什么是四轮 K（k=0~3）？**
因为做一次完整的内积，长度是 $K=4$。这就像是做一道工序需要 4 种零件，所以他要按顺序分 4 轮（k=0, k=1, k=2, k=3）来推进工作。

**3. 神奇的魔法发生在拿货（读取 Fragment）时：**
- **在第 1 轮（k=0）时：**
  这个工人去拿 $A$ 的第 0 列的前 4 个数字，抓在左手（`a_frag[0~3]`）；
  再去拿 $B$ 的第 0 行的前 4 个数字，抓在右手（`b_frag[0~3]`）。
  👉 **左手 4 个数，右手 4 个数，互相配对相乘（外积 Outer Product）**。$4 × 4 = 16$！他一口气往桌上的 16 个空盒子里各扔进去 1 个计算结果！
- **在第 2、3、4 轮（k=1, 2, 3）时：**
  他继续重复上面的动作。左手抓 4 个新的 $A$，右手抓 4 个新的 $B$，一相乘又是 16 个结果，扔进盒子里累加。

**4. 为什么 A Fragment 读取是 16？**
我们来算算这个超级工人总共拿了几次货：
- 他总共干了 4 轮（k=0, 1, 2, 3）。
- 每一轮，他拿了 4 个 $A$（抓在左手）和 4 个 $B$（抓在右手）。
- 所以，他拿 $A$ 的总次数是：**4 轮 × 每轮 4 个 = 16 次！**

| 项目 | 没有 Thread Tile | 引入 2D Thread Tile | 收益 |
|---|---|---|---|
| 所需工人（线程数） | 16 个 | **1 个** | 节约了调度成本，腾出了空间 |
| FMA 数学计算量 | 64 次 | 64 次 | 真正的数学计算量完全没变 |
| 去拿 $A$ 的总次数 | 64 次 | **16 次** | 🏆 **骤降到四分之一！** |
| 去拿 $B$ 的总次数 | 64 次 | **16 次** | 🏆 **骤降到四分之一！** |
| 每拿一对 A/B 的价值 | 产生 1 次计算 | **产生 16 次计算** | 数据价值被压榨到了极致 |

> 💡 **小白通俗解析：** 
> 总结一下：**Thread Tile 并没有减少数学计算，它消灭的是“重复跑腿”！** 
> 以前 16 个人每个人都要去拿面粉，同一袋面粉被拿了 16 次。现在 1 个人把 16 份面团全包了，他端一盆面粉过来，直接揉出 16 个小面团。这也就是所谓的**寄存器级数据复用**，是 2D Thread Tile 让性能瞬间起飞的根本原因！

### 【手工推导篇】傻瓜式的一步步演算验证

如果你还是觉得“左手拿 A，右手拿 B，一口气算出 16 个结果”有点像变魔术，那我们干脆**把规模缩小，亲手用笔算一遍，印证这个魔法！**

假设我们现在是一个超级工人，负责计算 C 矩阵中的一个 **2×2（即 4 个数字）** 的小方块。
假设内积长度 **$K=2$**（也就是说，我们只需干 2 轮）。

**【初始状态准备】**
我们需要的 $A$ 矩阵片段（左手要拿的 2 行）和 $B$ 矩阵片段（右手要拿的 2 列）如下：
```math
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
```
我们在工位上准备 4 个空盒子（也就是代码里的寄存器 `acc[2][2]`），初始状态全为 0：
```math
\text{acc} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
```

---

**【第 1 轮开始 (k = 0)】**
- **拿货**：
  - 左手拿 $A$ 的第 0 列：`a_frag = [1, 3]`
  - 右手拿 $B$ 的第 0 行：`b_frag = [5, 6]`
- **组装（外积计算）**：
  左手的每一个数，都要和右手的每一个数相乘，扔进对应的盒子里：
  - 左手第1个数(1) × 右手第1个数(5) = 5 → 扔进 `acc[0][0]`
  - 左手第1个数(1) × 右手第2个数(6) = 6 → 扔进 `acc[0][1]`
  - 左手第2个数(3) × 右手第1个数(5) = 15 → 扔进 `acc[1][0]`
  - 左手第2个数(3) × 右手第2个数(6) = 18 → 扔进 `acc[1][1]`
- **第 1 轮结束，盒子的状态变成了**：
```math
  \text{acc} = \begin{bmatrix} 5 & 6 \\ 15 & 18 \end{bmatrix}
```

---

**【第 2 轮开始 (k = 1)】**
- **拿货**：
  - 左手拿 $A$ 的第 1 列：`a_frag = [2, 4]`
  - 右手拿 $B$ 的第 1 行：`b_frag = [7, 8]`
- **组装（外积计算）并累加到之前的盒子里**：
  - 左手第1个数(2) × 右手第1个数(7) = 14 → 加上原来的 5，`acc[0][0]` 变成 **19**
  - 左手第1个数(2) × 右手第2个数(8) = 16 → 加上原来的 6，`acc[0][1]` 变成 **22**
  - 左手第2个数(4) × 右手第1个数(7) = 28 → 加上原来的 15，`acc[1][0]` 变成 **43**
  - 左手第2个数(4) × 右手第2个数(8) = 32 → 加上原来的 18，`acc[1][1]` 变成 **50**
- **第 2 轮结束，盒子的状态变成了**：
```math
  \text{acc} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

---

**【自我印证：我们的魔法算对了吗？】**
现在我们完全抛开外积魔法，用你大学学过的**最传统的矩阵乘法规则（行乘列）**来检查一下：
- $C_{0,0} = (A 的第 0 行) × (B 的第 0 列) = 1 × 5 + 2 × 7 = 5 + 14 = 19$ ✅
- $C_{0,1} = (A 的第 0 行) × (B 的第 1 列) = 1 × 6 + 2 × 8 = 6 + 16 = 22$ ✅
- $C_{1,0} = (A 的第 1 行) × (B 的第 0 列) = 3 × 5 + 4 × 7 = 15 + 28 = 43$ ✅
- $C_{1,1} = (A 的第 1 行) × (B 的第 1 列) = 3 × 6 + 4 × 8 = 18 + 32 = 50$ ✅

**结果完全一致，分毫不差！**

> 💡 **小白通俗总结：**
> 你看，我们只去了大仓库 **2 次**（$k=0$ 一次，$k=1$ 一次）。每次回来只是拿了 2 个 $A$ 和 2 个 $B$，总共搬了 **4 个数字** 的货，却瞬间完成了 **4 次有价值的乘法**。
> 这就是 2D Thread Tile 的核心数学原理：**把“行乘列”的内积计算，巧妙地转化为了沿着 K 方向一轮一轮进行的“外积相加”**。这也是 GPU 算矩阵乘法最底层、也是最强大的武器！


## Kernel 4：二维 Thread Tile，开始像小型矩阵乘法

Kernel 4 让一个线程负责 TM×TN 个输出，比如 4×4，一共 16 个 C。它会维护一个小的 accumulator 矩阵：

```cpp
acc[4][4] = 0
for k in 0..BK-1:
    a_frag[4] = A 的 4 个行方向值
    b_frag[4] = B 的 4 个列方向值
    for m in 0..3:
        for n in 0..3:
            acc[m][n] += a_frag[m] * b_frag[n]
```

> 💡 **小白通俗解析：** 这一步非常重要。一个线程手里同时有 A 的一小列片段和 B 的一小行片段，然后做 outer product。4 个 A 值和 4 个 B 值组合出 16 次乘加。复用率显著提高。

> 💡 **小白通俗解析：** 这就是为什么 TM/TN 不是装饰参数。TM/TN 越大，一个线程负责的 C 子块越大，寄存器复用越强；但寄存器占用也越高，太大可能降低 occupancy。

## Kernel 5：寄存器缓存，把最热的数据放到手边

SMEM 已经比 GMEM 快，但 SMEM 也不是免费。Kernel 5 把当前 K 步要用的 A/B fragment 先读到寄存器。

```cpp
for k in 0..BK-1:
    a_frag[m] = As[row+m, k]
    b_frag[n] = Bs[k, col+n]
    acc[m][n] += a_frag[m] * b_frag[n]
```

> 💡 **小白通俗解析：** Registers 是线程私有且最快的存储。把 a_frag 和 b_frag 放到寄存器后，内层循环可以少访问 SMEM，多用寄存器完成乘加。

> 💡 **小白通俗解析：** SMEM 像灶台旁的货架，寄存器像厨师已经抓在手里的调料。最常用的一小撮东西，当然放手里最快。

## Kernel 6：FLOAT4，一次搬四个连续数字

如果内存地址连续且对齐，可以用 `float4` 一次读取 4 个 FP32。它的直觉是减少访存指令和事务数量。

```cpp
普通读取：
load A[x]
load A[x+1]
load A[x+2]
load A[x+3]

FLOAT4：
load float4(A[x..x+3])
```

> 💡 **小白通俗解析：** FLOAT4 的收益建立在“数据连续、对齐、马上会用”上。它不是为了让数学变少，而是让搬运动作更粗粒度、更高效。

> 💡 **小白通俗解析：** 像买鸡蛋。一次拿一个鸡蛋要跑四趟，一次拿一盒四个只跑一趟。但前提是你确实要这四个，而且盒子放得整齐。

## Kernel 7：双缓存，边做饭边备下一盘菜

双缓存的目标是把“搬下一块 tile”和“计算当前 tile”重叠起来。通常会准备两份 SMEM buffer：

```cpp
buffer 0: 当前正在计算的数据
buffer 1: 正在预取的下一轮数据

下一轮交换：
buffer 1 变成当前计算
buffer 0 用来预取再下一轮
```

> 💡 **小白通俗解析：** 如果计算当前 tile 时，硬件同时在准备下一 tile，那么 GMEM 延迟就能被部分隐藏。这样 pipeline 更满，计算单元更不容易等数据。

> 💡 **小白通俗解析：** 厨师炒第一盘时，助手已经把第二盘菜洗好切好。等第一盘出锅，第二盘马上下锅，中间不用停下来等备菜。

## 把 BM、BN、BK、TM、TN 讲成人话

| 参数 | 第一性原理含义 | 调大可能带来什么 | 调太大有什么风险 |
|---|---|---|---|
| BM | C tile 的行数 | A 数据复用更多 | SMEM/寄存器压力上升 |
| BN | C tile 的列数 | B 数据复用更多 | 线程组织更复杂 |
| BK | 每轮 K 方向深度 | 每轮计算更多 | SMEM 占用增大 |
| TM | 单线程负责的 C 行数 | 一个 B 值复用更多 | accumulator 变多 |
| TN | 单线程负责的 C 列数 | 一个 A 值复用更多 | 寄存器占用变高 |

> 💡 **小白通俗解析：** 这些参数没有绝对最优。它们是在“复用更多”和“资源够不够”之间找平衡。高性能 SGEMM 不是把每个参数都调大，而是让它们共同适配某一代 GPU。

## 最终总结：七个 kernel 的学习价值

- Kernel 1 让你看懂一个线程如何从 A 行和 B 列算 C。

- Kernel 2 让你看懂 block 内线程如何共享 A/B tile。

- Kernel 3/4 让你看懂 thread tile 如何提高寄存器级复用。

- Kernel 5 让你看懂为什么 fragment 要进寄存器。

- Kernel 6 让你看懂连续对齐访问为什么适合向量化。

- Kernel 7 让你看懂为什么高性能 kernel 要做流水线。

> 💡 **小白通俗解析：** 
读完请自测：如果 TM=4、TN=4，一个线程负责几个 C 输出？如果 BK 变大，K-loop 轮数会怎么变？如果 FLOAT4 读取的数据不连续，还能得到同样收益吗？

---

![](images/head.png)

![](https://img.shields.io/badge/build-passing-brightgreen) ![](https://img.shields.io/badge/ubuntu-18.04-blue) ![](https://img.shields.io/badge/cuda-10.2-blue) ![](https://img.shields.io/badge/nvidia-RTX3090-blue) ![](https://img.shields.io/badge/cmake-3.21-blue)



# 概述

面向NVIDIA GPU，使用CUDA编程逐步优化矩阵乘法运算性能：

| 核函数   | 描述                    | GFLOPS   | 自定义核函数/CUBLAS（%） |
| -------- | ----------------------- | -------- | ------------------------ |
| CUBLAS   | 官方库函数              | 14448.69 | 基准                     |
| kernel_1 | 朴素实现                | 2262.168 | 15.65657                 |
| kernel_2 | 共享内存缓存            | 4216.536 | 29.18283                 |
| kernel_3 | 一维Thread Tile并行优化 | 7809.629 | 54.05078                 |
| kernel_4 | 二维Thread Tile并行优化 | 12251.3  | 84.79179                 |
| kernel_5 | 寄存器缓存              | 12177.95 | 84.28412                 |
| kernel_6 | FLOAT4向量访存          | 13161.49 | 91.09125                 |
| kernel_7 | 双缓存预取              | 13634.98 | 94.36832                 |

> NVIDIA GeForce RTX 3090，矩阵尺寸5120

# 配置

- 编译采用 `gcc 7.5.0` under Ubuntu 18.04.5 LTS
- NVIDIA CUDA version: `CUDA 10.2`；

# 目录

```
NVIDIA_SGEMM_PRACTICE                                   # 根目录
    ├── images                                          # 图片结果
    │     ├── describe_kernel_1.png  
    │     ├── describe_kernel_x.png
    │     └── kernel_x_vs_y.png
    ├── test                                            # 测试结果
    │     ├── test_kernel_0.txt 
    │     ├── test_kernel_1.txt 
    │     └── test_kernel_x.txt 
    └── src                                             # 源文件
    │    ├── kernel
    │    │  ├── kernel_1.cuh                            # 声明和定义
    │    │  ├── kernel_2.cuh
    │    │  └── kernel_x.cuh
    │    ├── kernel.cuh
    │    ├── utils.cuh                                  # 辅助函数
    │    └── utils.cu
    ├── plot.py                                         # 根据test结果绘图
    ├── run.sh                                          # 运行编译后可执行文件
    ├── sgemm.cu                                        # 主程序
    └── CMakeLists.txt                                  # 编译相关
```

# 运行
1. 配置NVCC编译参数
> 在CMakeLists.txt中修改`set(CUDA_NVCC_FLAGS -arch=compute_70;-code=compute_70)`
2. 配置矩阵计算最大尺寸
> 在`sgemm.cu:16`中修改`size_len`，建议初次运行设置为16，过大尺寸可能导致电源超负荷主机重启；
3. 编译
`cd build && cmake .. && make`
4. 运行run.sh，统计各个核函数计算效率，结果保存在test目录；
5. 计算效率折线绘图

> `python plot.py 0 1`表示绘制CUBLAS和kernel_1计算效率对比图；

# 逐步优化

##  kernel 1 

**Naive基础版矩阵乘法实现**

将每个逻辑线程与矩阵C的每一个元素相对应，每个线程负责C中一个元素的计算；

![](./images/describe_kernel_1.png)

```cpp
__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // 全局x
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // 全局y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx]; // 两次全局内存访问和一次FMA（累加乘）
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}
```

![](./images/kernel_culas_vs_1.png)

未经过优化的矩阵乘法性能不足CUBLAS的1/10，具体分析如下；

- 计算访存比：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2；
- 访存量：访问全局内存，C矩阵每个元素计算需要访问`2K`个单精度浮点数，完成全部计算需要` 2*K*M*N`；

全局内存访问延迟高（几百cycle），同时相同位置元素被重复读取（C中同一行元素计算共享A中同一行元素，C中同一列元素计算共享B中同一列元素），另一方面，较低的计算访存比无法有效隐藏访存延迟，因此，访存延迟和计算访存比是导致kernel 1效率低下的原因。

## kernel 2

**利用共享内存缓存减少全局内存访存量和访存延迟**

访存延迟来自于全局内存的高延迟和全局内存的重复访问。共享内存是片上内存，具有较低的访存延迟（几十cycle）,使用共享内存进行缓存可降低访存延迟；

![](./images/describe_kernel_2.png)

> BM和BN表示block tile的高和宽，BK表示待缓存的全局内存的步长，即一个block的计算需要缓存K/BK次；

共享内存缓存全局内存A tile和B tile，完成C block中所有元素的FMA计算，不断滑动缓存区域，更新block；

```cpp
/*
dim3 blockDim(1024);
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/

template<const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // 申请共享内存空间
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {
        // 缓存A_tile和B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}
```

![](./images/kernel_1_vs_2.png)

- 访存量：每个block需要从global memory中读取`(K/BK)*(BM*BK+BK*BN)`个单精度浮点数，整个C存在`(M/BM)*(N/BN)`个block，因此完成C中所有元素计算需要读取`(M/BM)*(N/BN)*(K/BK)*(BM*BK+BK*BN)`个单精度浮点数

kernel 1受限于全局内存的访存延迟和重复访问，优化前全局访存量为`2*K*M*N`，共享内存缓存优化后，访存量减少为原来的`1/2*(1/BN)*(1/BM)`,当`BN=BM=32`时，访存减少至1/32；另一方面shared memory访存延迟远低于全局内存，因此计算效率得到了一定程度的提升。

## kernel 3

**利用一维thread tile优化**

已知可以通过增加block大小（BM，BN）值，进一步降低全局内存的访问量，因此将BM和BN从32提升至64；

> **是否能通过无限增加block size降低全局访存？**
>
> 不能，一方面，block分块矩阵尺寸过大，block数量减少，这样会造成大量 SM（Streaming Multiprocessor）的闲置浪费；另一方面，BN和BM的增加，需要申请更多的共享内存，单线程内共享内存占用越多，活跃线程束越少，不利于隐藏指令延迟；

因此，在增加BM和BN值的同时，为了减少共享内存占用，一方面减小BK值，降低为8；

> 当增加block size时，应尤其注意共享内存的消耗，限制共享内存尺寸和block中线程的数量，避免因资源不足无法启动核函数

![](./images/describe_kernel_3_1.png)

另一方面，通过共享内存缓存减少了全局内存访存量和FMA乘累加的访存延迟，但计算访存比没有得到改善，每次迭代计算都需要两个访存指令和一个计算指令，因此，引入thread tile，即一个线程负责block中多个元素的计算，TM和TN分别表示thread tile的高和宽。

![](./images/describe_kernel_3_2.png)

```cpp
/*
dim3 blockDim(512);
dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
mysgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/


template<const int BM,
        const int BN,
        const int BK,
        const int TM>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = BM * BN / TM; // 一个线程负责block中计算TM个元素

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM + 1] = {0.}; // 每个线程负责TM个元素，则需要申请TM个寄存器保存累加值，额外的一个寄存器用于缓存；
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            tmp[TM] = Bs[tx + i * BN]; // 额外的一个寄存器，避免反复从共享内存中读取Bs[tx + i * BN]
            #pragma unroll  // 循环展开，增加指令并行度
            for (int j = 0; j < TM; j++) {
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
    }
}
```

![](./images/kernel_2_vs_3.png)

本例从两方面进行优化：

- 全局内存访存量：相比于初始版本，通过对`64*64`block size进行缓存，访存量降至1/64；
- 计算访存比：引入thread tile，利用单个线程负责多个元素计算，增加计算访存比；当TM=8时，每执行共享内存As的8个次访存指令和共享内存Bs的1个访存指令，可执行8次计算指令，相比初始版本的计算访存比1:2，提高至8:9，有效隐藏访存延迟；

通过本例的两方面优化，矩阵乘法计算效率显著提高近一倍；

## kernel 4

**利用二维thread tile优化**

将thread tile设置为二维，即一个线程负责一小块元素的计算，从而进一步增加block尺寸，减少全局访存数量；

>  增加thread tile尺寸，可以在相同的线程数量或更少的线程数量下，计算更大的block size;

更重要的是，单线程负责计算更多的C元素区域，可以增加指令级并行程度；

> 为什么可以提高指令并行程度？
>
> 单线程处理的指令数量越多，流水线级越长，由于单线程流水线可并行处理多条指令，虽然单条指令执行变慢，但单位时间内处理的指令数量变多，提高了吞吐量，隐藏指令延迟；指令级并发相比与线程级并发更具优势。

![](./images/describe_kernel_4.png)

设置一个线程负责8×8区域内元素计算，即thread tile=8×8，TM=8,TN=8；

```cpp
// BM=BN=128，BK=8，TM=TN=8，共享内存大小128*8
dim3 blockDim(256);
dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;  // 128*8/256=4，需要所有线程搬运4轮，可将全局内存中128*8大小区域搬运至共享内存

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

// 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值；
float tmp[TM][TN] = {0.}; 

// 单个线程循环TM，TN完成thread tile内元素的乘累加
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
}
```

全局访存量：相比未引入共享内存缓存版本，全局内存访存量减少至`1/2*(1/BM+1/BN)=1/128`,访存量显著降低。

![](./images/kernel_3_vs_4.png)

实际测试发现，相比与一维thread tile，由于二维thread tile进一步降低了全局访存量、提升计算访存比，矩阵乘法效率显著提升一倍。

## kernel 5

**寄存器缓存共享内存**

![](./images/describe_kernel_5.png)

由下方代码可知，单个线程计算thread tile元素乘累加时，共享内存会被重复访问。

```cpp
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];  //内层循环中 As[(ty + j) * BK + i] 重复访问TN次
}
```

共享内存相比全局内存能够大大减少访存延迟，但共享内存延迟（几十cycle）相比于计算延迟（几cycle）仍然较大，因此，采用寄存器对共享内存As、Bs进行缓存，避免共享内存的重复访问；

```cpp
float a_frag[TM] = {0.};
float b_frag[TN] = {0.};

for (int i = 0; i < BK; i++) {
    for (int j = 0; j < TM; j++) {
        a_frag[j] = As[(ty + j) * BK + i];     // 采用a_frag寄存器数组缓存thread tile所需的As共享内存数据；
    }
    for (int l = 0; l < TN; l++) {
        b_frag[l] = Bs[tx + l + i * BN];       // 采用b_frag寄存器数组缓存thread tile所需的Bs共享内存数据；
    }
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            tmp[j][l] += a_frag[j] * b_frag[l];
    }
}
```

当TM=TN=8时，经过寄存器缓存，每个thread tile需要执行8个As共享内存访存指令和8个Bs共享内存访存指令，可进行8×8=64个计算指令，计算访存比相比于初始版本的1/2提升至64:16，可有效隐藏访存延迟；

![](./images/kernel_4_vs_5.png)

实际测试发现，经寄存器缓存实际性能并未发生明显变化，原因可能是当前性能瓶颈并非共享内存的重复访问；

## kernel 6

**向量内存指令FLOAT4优化**

- 计算指令：GPU是以4维向量为基本单位进行计算的，4个浮点数组成的float4向量是GPU最基本的类型，使用GPU对两个float4进行向量计算与对两个整数或两个浮点数进行计算一样，只需要一个指令即可完成；
- 内存指令：与发出单个指令生成单独的内存事务获取相同数量的字节相比，通过向量内存指令所需的内存事务更少，减少了内存控制器的争用；另一方面，使用矢量加载每个字节需要更少的索引计算；

![](./images/describe_kernel_6.png)

例如，BM=128，BK=8，线程数量为256，若每个线程每次取1个浮点数，每个线程需要消耗4次内存指令，才能将全局内存搬运至共享内存，若采用float4向量内存指令，每个线程每次可以搬运4个浮点数，则每个线程仅需要执行一次内存指令即可完成搬运。

关键代码示例如下：

```cpp
#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float ldg_a_reg[4 * ldg_a_num] = {0.}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置As矩阵

//  共享内存缓存全局内存
for (int i = 0; i < BM; i += a_tile_stride) {
    int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
    FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
            FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
    // As转置存，其中ldg_a_reg做中间缓存，目的是读取时可以按FLOAT4读取
    As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
    As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
    As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
    As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
}

for (int i = 0; i < BK; i += b_tile_stride) {
    FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
        FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // 不需要转置
}


// 寄存器缓存共享内存
// ty,tx为当前线程对应thread tile的左上角元素在block中的位置
#pragma unroll
for (int m = 0; m < TM; m += 4) {
    FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]); // 偏移到当前thread tile
}
#pragma unroll
for (int n = 0; n < TN; n += 4) {
    FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]); // 偏移到当前thread tile
}
```

全局内存无法直接写入共享内存，需要寄存器做中介，其中As写入将全局内存->将寄存器->共享内存过程显示的描述出来，而Bs写入并不是不需要寄存器参与，只是编译器隐藏了这段代码；As缓存显示运用寄存器的目的在于将As进行转置，转置前的一列在转置后变成一行，内存连续，便于float4读取；

![kernel_1](./images/kernel_5_vs_6.png)

实际测试，整体计算效率增加；

## kernel 7

**数据预取**

单缓存是指申请单块共享内存，缓存全局数据，申请单块寄存器内存，缓存共享数据，单块缓存不能实现读取和存储并行进行，因为数据之间存在依赖。例如单缓存场景，计算依赖共享内存数据，为保证计算前全局内存完全存入共享内存，需要进行一次同步；同样因为计算依赖共享内存数据，所以在存新一轮全局内存到共享内存前也需要进行一次同步，保证上一轮计算完成。

双缓存通过申请双倍存储空间，将读和写分开，计算数据读取一块存储空间同时，可以同时向另一块内存写入下一轮依赖的数据，因此，只需要保证计算前待读取共享内存完成写入，即一次同步即可。

> 双缓存使读写同步进行，实现数据预取，隐藏内存延迟。

![](./images/describe_kernel_7.png)

![](./images/kernel_6_vs_7.png)

采用双缓存技术实现数据预取，计算效率得到了进一步提升；

![](./images/kernel_culas_vs_7.png)

基本可以接近CUBLAS官方矩阵乘法的计算效率；

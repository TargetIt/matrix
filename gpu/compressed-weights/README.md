# GPU 压缩权重：Arm Mali G2-Ultra NX 与高通、NVIDIA、Apple

**Compressed weights 的系统价值，是让计算单元少等数据、内存少搬数据。** 模型文件变小只是第一步；权重在运行时仍保持压缩，并在接近计算的位置还原，才可能持续节省带宽和功耗。

Arm 已在 G2-Ultra NX 官方架构图确认这项能力，但没有在已核查的公开材料中披露编码格式、压缩率、解码器位置或稀疏计算倍率。理解其来龙去脉，最可靠的路线是：先看 NX 原始发布，再看 Arm 历史硬件与 Vela 编码器，最后区分当前模型/API 能证明什么。

本文证据截止 **2026-09-09**。正文标明“已确认”“推断”和“未公开”；源码链接尽量固定到核查版本。这里只讨论公开证据与系统分析，没有 NX 真机压缩性能实测。

阅读导航：[Arm 发布证据](#1-arm-到底公布了什么) · [压缩的不同含义](#2-先分清四种压缩) · [Arm 历史与源码](#3-arm-的来龙去脉) · [运行时系统链路](#4-从模型到硬件系统链路怎么理解) · [厂商对照](#5-高通nvidiaapple-分别怎么做) · [工程判断](#6-对移动神经图形的实际意义) · [来源索引](#7-一手资料与源码索引)

## 1. Arm 到底公布了什么

2026-09-08 的 [G2-Ultra NX 官方发布文章][a1]包含一张 **Neural Accelerator: Hardware architecture** 图。[点击直接看原图][a2]，其 Performance 栏明确列出：

> INT8 and INT16 · Whole tensor operations · Compressed weights

图中把 Neural Accelerator 放在 shader core 内，与 Execution Engine 共享 Memory System 和 Control Structures；正文说明复用 GPU 一致性缓存。这是 **NX 硬件支持压缩权重**的直接证据，不只是模型训练工具宣称“支持压缩”。

但这张图不能回答以下问题：

| 问题 | 公开证据的边界 |
| --- | --- |
| 压缩的是 INT8 还是 INT16 权重？ | 两类处理精度与压缩权重并列，不能据此确定所有权重格式组合 |
| 无损编码，还是低比特有损表示？ | 图中未说明；不能直接写成 Huffman、INT4、ASTC 或某种稀疏格式 |
| 是否复用 Ethos 的编码器？ | 没有找到公开确认；技术思路可对照，码流兼容性未知 |
| 解压在 L2 前后、L1 后，还是 MAC 入口？ | 未披露；共享缓存不等于压缩状态贯穿每一级缓存 |
| 是否能跳过零权重计算、吞吐翻倍？ | 未披露；权重压缩不能自动推出稀疏计算加速 |
| 压缩率与独立收益是多少？ | 未披露；发布中的整套神经图形收益不能归因于此单项 |

**可成立的系统推断：** 将它列为神经加速器硬件性能特性，说明目标不止于缩小下载包，而是改善执行时的权重供给；具体由什么单元、以什么粒度解码，仍需 TRM、驱动说明或实测补齐。

## 2. 先分清四种“压缩”

权重是训练得到、推理时通常不变的参数；activation 是每次输入产生的中间结果。两者的压缩时机与复用方式不同。

| 方法 | 通俗解释 | 是否改变数值 | 是否必然减少计算 |
| --- | --- | --- | --- |
| 低比特量化 | 每个数用更少位存，例如 FP16 → INT4，并附比例尺 | 通常有误差，需要校准或训练 | 否；可能先还原成 FP16 再计算 |
| 聚类／查表 | 很多权重共用一张“小字典”，只存词条编号 | 聚类可能有误差；之后查表可精确恢复字典值 | 否 |
| 稀疏编码 | 零不逐个存，保存非零值及位置 | 编码本身可无损；把非零剪成零可能损失精度 | 只有匹配的稀疏执行硬件才可能少算 |
| 无损熵编码 | 常见数值用短码，连续零用长度表示 | 对送入编码器的数值无损 | 通常只改变存储与搬运 |

这些方法可以叠加。**“先量化／剪枝，后无损编码”不代表整个过程无损**：前一步可能改变模型，后一步只保证原样还原已经处理过的权重。Arm 的 [Ethos-U 官方说明][a4]明确区分了这两层。

用一个不代表 NX 格式的小例子：8 个 INT8 权重 `[0, 0, 5, 0, 0, 0, -2, 5]` 原本占 64 bit；若用 8 bit 零值掩码，加上三个非零值的 24 bit，只需 32 bit，另计块头及对齐。这说明“仍按 INT8 算”和“权重被压缩”完全可以同时成立。实际收益由数据分布、元数据和分块共同决定。

## 3. Arm 的来龙去脉

### 3.1 2018：压缩权重已经进入 Arm ML Processor 的数据通路

Ian Bratt 的 [Hot Chips 30 官方演讲，第 13、17–20 页][a3]给出了完整因果链：权重离线编译为专用格式，在内部 SRAM 中仍保持压缩，读出后经 Weight Decoder 送入 MAC。剪枝增加零值，聚类增加重复值，从而提高编码效率。

这条路线解决的是“芯片算得快，但喂数太贵”。演讲同时介绍 activation 压缩和 tiling；后者通过把工作集留在 SRAM，减少中间结果往返 DRAM。**压缩与数据复用从一开始就是配套设计。**

演讲讨论的是当年的 ML Processor，不是 Mali NX。它证明 Arm 早有这套系统设计经验，不能证明 G2 原样复用了该硬件。

### 3.2 Ethos-U／Vela：可以读到真正的编码器

[Arm 2023 年 pruning/clustering 技术文章][a4]明确说明：Vela 按硬件需要排列权重块，再做无损压缩；Ethos-U 的专用 weight decoder 在执行时还原。模型中的连续零、重复权重越容易编码，访存等待就越可能下降。

可从下面三处直接读源码，固定于 Vela `a43300cb989c`：

- [`weight_compressor.py → encode_weights()`][a5]：依据卷积布局、block traversal、输入位宽和 accelerator 配置，调用 `mlw_codec.reorder_encode()`。说明压缩必须配合硬件消费顺序。
- [`mlw_encode.c`][a6]：包含 palette、zero-run 和 GRC 变长编码路径，以及 slice header、直接编码选项。GRC 路径用商／余数和 unary 编码，可理解为 Golomb–Rice 类编码；不是简单 ZIP。
- [`mlw_decode.c`][a7]：按 palette 或直接值恢复数值，再把连续零插回。这里展示的是可读的软件参考解码，不是公开 NX RTL。

需要注意，源码里的 **palette 编码**不等于编码器自动做有损聚类：前者给已有数值编号，后者才是训练阶段主动把相近值合并。

到 [2024 年 Ethos-U85 发布][a8]，Arm 把“standard/fast weight decoder”和“原生 2:4 sparsity、双倍吞吐”分别列出。这个区分很重要：**解压得快、存得少、零乘法少算，是三个不同设计维度。** U85 能做的，不能直接填进 NX 的规格表。

### 3.3 2025–2026：神经图形先把模型做成适合移动硬件的样子

[2025 NSS 官方演讲][a9]第 19 页把 INT8、共享带宽、功耗列为设计约束。其方法不是让低精度网络直接生成最终 HDR 颜色，而是让网络预测重建参数，由后处理完成图像重建；这降低了量化的难度。

[2026 NFRU 官方演讲][a10]的 **Quantized Network** 页进一步明确：INT8 用于降低带宽并发挥 NX 吞吐，浮点候选颜色保留，网络预测混合系数。该页的幻灯片编号是 55，含讲稿 PDF 的物理页是 74。

这两份 Hugging Face 文件虽然模型卡称作 paper，实际是**带讲稿的技术演讲材料**。它们解释算法与精度选择，没有给出 NX 权重压缩码流。

### 3.4 GitHub／Hugging Face 的证据到哪一层

本次核查了 Model Gym、VGF Library 和 Vulkan Emulation Layer 的源码，并读取 NSS/NFRU 模型卡与文件清单。

| 开放组件 | 实际看到的内容 | 不能由此推出的结论 |
| --- | --- | --- |
| [Model Gym 导出代码][a11] | `FP32`、`QAT_INT8`、`PTQ_INT8`；TOSA quantizer、`convert_pt2e`、`VgfPartitioner` | INT8 导出不等于已公开 NX 压缩编码器 |
| [NSS 文件清单][a12]／[NFRU 文件清单][a13] | FP32/QAT 检查点、INT8 VGF、量化元数据、shader 与场景 | `.pt`／`.vgf` 文件大小不能直接当成硬件驻留权重大小 |
| [VGF `AddConstant()`][a14] | 拷贝常量字节，记录大小、位置和稀疏维度等元数据 | 容器的 encoder/decoder 名称不是硬件权重 codec 的证据 |
| [Vulkan Emulation Layer][a15] | 将图算子映射到 Vulkan compute 的实现 | 在普通 GPU 上运行成功，不代表模拟了 NX 解码吞吐、缓存或功耗 |

另一个容易误读的地方：NSS 源码里的 [`use_sparse_filter_2x2` 与 `SPARSE_PRUNED_CHANNEL_MAP`][a16]描述上采样后处理的采样点／通道选择。它不是“CNN 静态权重已按 2:4 剪枝”的证据。网络预测的逐像素滤波权重，也不能与离线训练权重混为一谈。

**结论：公开软件链足以理解模型、量化、图提交；目前尚不能从这些材料复原 NX 私有压缩权重的二进制格式和完整解码通路。**

## 4. 从模型到硬件，系统链路怎么理解

下面是通用的“压缩驻留、按需解码”设计，不是 NX 的内部框图：

```mermaid
flowchart LR
    A[训练与校准后的权重] --> B[编译器分块、重排、编码]
    B --> C[压缩权重与元数据驻留内存]
    C --> D[按块搬运与片上缓存]
    D --> E[解码、查表或反量化]
    E --> F[乘加单元]
```

**训练端**决定权重是否适合低比特或稀疏表示；**编译器**决定编码、布局、分块与调度；**运行时和硬件**决定是否真的以压缩形态搬运、何时解码。这三层都匹配，才会有端到端收益。

Arm Vulkan API 为这种分工留下了空间。[Graph Constants 官方示例][a17]在创建 data-graph pipeline 时提交权重的 host pointer；[`VkDataGraphPipelineConstantARM`][a18]描述的是逻辑张量及其布局，应用不必提供 NX 私有码流。**据此推断，后端在 pipeline 准备阶段做目标专用重排／编码是自然的实现位置；但这不是 NX 编译时机的官方确认。**

API 还有一个值得留意的[半结构化稀疏提示][a19]：`dimension`、`zeroCount`、`groupSize`。但规范明确允许实现忽略提示、按 dense 处理，且该提示**不改变上传数据的布局**。因此“API 能描述 2:4”仍不能证明“NX 以 2:4 加速”。

解压位置决定到底省什么：

- **加载时整份展开到 DRAM：** 主要省下载／磁盘空间，后续推理仍搬完整权重。
- **从 DRAM 压缩读取，片上还原：** 能省 DRAM 带宽，但是否节省缓存容量取决于还原位置。
- **压缩驻留到局部 SRAM，消费时还原：** 连片上存储也能获益；Arm 2018 年演讲明确展示过这条路径。
- **算术单元直接消费紧凑表示：** 例如稀疏 Tensor Core 选择对应 activation，可能进一步减少乘法。

权重通常不随每帧输入改变，所以较复杂的编码工作可以离线完成；解码则要持续供数。工程上需要同时处理块寻址、对齐、最差压缩率、解码吞吐和缓存竞争。**压缩得更小，如果解码来不及，仍可能更慢。**

## 5. 高通、NVIDIA、Apple 分别怎么做

### 5.1 高通：低比特硬件、LPBQ 与软件流式解码是三条线

**移动端硬件。** Snapdragon 8 Gen 2 于 2022 年引入 Hexagon INT4 支持；[官方技术说明][q1]同时介绍 micro-tile inferencing 与 GPU/ISP direct link。INT4 减少表示位数，micro-tile 减少层间 activation 往返，direct link 降低跨单元搬运成本；三者作用不同。这些是 Hexagon／SoC 证据，不能写成 Adreno shader core 具备 NX 同款 weight decoder。

**最贴近“压缩存、较宽位数算”的开放实现是 LPBQ。** [AIMET 2.19 文档][q2]给出 `bitwidth=4, decompressed_bw=8, block_size=64` 的例子，把低位宽分块编码调整到共同的高位宽逐通道网格，以复用已有 per-channel kernels。

[`GroupedBlockQuantizeDequantize` 源码][q3]进一步写出比例关系：`scale = per_channel_scale × per_block_int_scale`。通俗地说：每块保存小整数权重及整数倍率，再与通道比例尺组合，避免每块都独立携带较大的浮点比例尺。**它公开了量化／重建规则，不等于公开了 Hexagon 的物理解码位置。**

部署证据可以看 [AI Hub Qwen3-4B 导出说明][q4]与 [Qualcomm Hugging Face 模型卡][q5]。这里的 W4A16 表示权重 INT4、activation INT16，不能机械套用其他框架中 A16=FP16 的含义，见 [AI Hub API 定义][q6]。

**云端还有明确的软件解码实例。** [Cloud AI 100 官方文章，2024-01][q7]说明：编译时把 FP16/FP32 权重转为 MXFP6，运行时由 vector engine 的软件 kernel 即时解压，随后仍按 FP16 计算，并将搬运、解压、计算重叠。官方报告权重存储最多节省约 61%。这是存储精度与计算精度分开的实例，但不是移动 Adreno 的硬件特性。

### 5.2 NVIDIA：从稀疏编码到 Tensor Core 内联查表

**NVDLA：可直接读 RTL 的传统压缩权重路径。** [官方格式文档][n1]规定三份数据：非零权重、逐元素零值掩码 WMB、每组压缩字节数 WGS；[`NV_NVDLA_CSC_WL_dec.v`][n2]提供解码逻辑。它很适合研究“格式—访存—解码”的接口，但 NVDLA 是 DLA IP，不能等同于 GeForce/CUDA GPU；公开 RTL 也不保证每套软件发行版都完整启用压缩。

**Ampere/A100，2020：2:4 Sparse Tensor Core。** 每四个权重至多两个非零，存非零值和位置元数据；Tensor Core 依位置选择另一操作数中对应的 activation，只做必要乘法。[官方原理图][n3]与 [cuSPARSELt 示例][n4]把这条路径讲得很清楚。它同时省权重带宽和算术工作，但前提是模型满足约束并选择了 sparse kernel；不是任意零值分布都能翻倍。

**TensorRT 10，2024：INT4 Weight-Only Quantization。** [官方说明][n5]指出 kernel 从内存读 INT4，反量化后按较高精度做点积。这主要解决带宽／容量，不能把 4 位存储直接算成 4 位矩阵计算峰值。

**Blackwell：NVFP4。** [官方格式说明][n6]是 4 位 E2M1 值，加每 16 个元素共享的 FP8 scale，并有逐 tensor scale；配套 Tensor Core 支持低精度运算和硬件 scaling。[CUTLASS 示例][n7]可看具体 GEMM。只计元素与块 scale，平均已是 `4 + 8/16 = 4.5 bit/weight`，还没算全局 scale、对齐及其他模型数据。

**最新 Rubin：3 位 LUT 权重。** [2026 年 Rubin 官方架构文章][n8]明确说明 matrix B 可保存 3 位查表索引，Tensor Core 内联解析，减少权重存储和搬运。这比泛称“支持 FP4”更接近硬件直接消费压缩权重。该文另述 attention activation 的自适应稀疏压缩；两者不是同一个特性。文章没有完整公开 LUT 分组、码表开销和通用部署配方，不能假设所有权重总成本恰好为 3 bit。

### 5.3 Apple：ANE 的按需还原，以及 GPU 复用 ASTC

**Core ML／ANE：从小文件走向小运行时权重。** [WWDC23 官方讲解][p1]对比了 iOS 16 的提前展开和 iOS 17 部分场景下的即时解压。palettization 用低位宽索引引用权重码表；较小的内存读取可能改善带宽受限层的延迟。

[Core ML Tools 官方指南][p2]强调，是否即时还原取决于芯片、计算后端及模型；不能只看压缩文件就判断运行时收益。[开源 MIL 算子][p3]也明确允许加载时或运行时还原，包含 `constexpr_blockwise_shift_scale`、`constexpr_lut_to_dense`、`constexpr_sparse_to_dense`。所以看到 `constexpr` 不应断言一定在加载时全量展开。

[Apple 官方 Stable Diffusion 仓库][p4]提供混合位宽 palettization 的实际用法；[palettization 指南][p5]补充了分组码表、vector palettization 等细节。这些工具公开了表示方法，不是 ANE RTL。

**Apple GPU：2025 年已公开用 ASTC 压缩服务器模型权重。** [Apple 官方技术报告第 6 节][p6]给出了具体实现：

1. 每 6×6、即 36 个权重编码为一个 128 bit ASTC 块。
2. 为适应 HDR-ch 非负值表示，先减去该块最小值，并另存一个 FP16 最小值。
3. GPU 固定功能 ASTC 单元按需解码，在张量运算中融合加回偏移；shader 无需承担一整套软件解压。
4. ASTC 属于有损压缩，配合低秩适配和质量恢复处理。

报告的约 3.56 bit/weight 是 `128/36` 的块负载口径。**若把每块的 FP16 最小值计入，按上述描述复算就是 `(128+16)/36 = 4 bit/weight`**，还未计适配器和对齐。“不占 shader 解压算力”也不意味着硬件不耗能或访存无限快。

这个具体部署用于 **AFM server 模型**，不能写成 iPhone 端 2 位模型、MetalFX 或所有 Core ML 模型都采用 ASTC。它证明的系统思路非常实用：已有纹理解码硬件也可以服务模型权重；但 Mali 支持 ASTC 并不能反证 NX compressed weights 就是 ASTC。

### 5.4 放在同一张表看

| 实现 | 压缩表示 | 哪里还原／消费 | 最直接的收益 | 证据边界 |
| --- | --- | --- | --- | --- |
| Mali G2-Ultra NX | 未公开 | NX 硬件能力已确认，具体位置未知 | 权重供给优化是合理推断 | 不能填入 Ethos 编码率或 2:4 倍率 |
| Arm Ethos／Vela | 零游程、palette、变长编码等 | 专用 weight decoder | 存储与搬运 | 是历史与旁系实现 |
| Qualcomm LPBQ | 低位权重＋分块倍率＋通道 scale | 后端重建到共同高位宽网格 | 紧凑表示、复用 per-channel kernel | 物理解码位置未由工具证明 |
| Cloud AI 100 MXFP6 | 带分块 scale 的低位浮点 | vector 软件 kernel → FP16 | DRAM 容量和带宽 | 云端实例，非 Adreno |
| NVIDIA Ampere 2:4 | 非零值＋位置 | Sparse Tensor Core 选择 activation | 带宽与有效计算吞吐 | 需结构约束和 sparse kernel |
| NVIDIA Rubin LUT | 3 位索引＋码表 | Tensor Core 内联查表 | 权重容量和数据移动 | 码表开销、部署细节仍需更多资料 |
| Apple Core ML／ANE | palette／量化／稀疏表示 | 依模型和平台提前或即时还原 | 部分场景的内存和延迟 | 小文件不保证低运行内存 |
| Apple GPU AFM server | ASTC 块＋FP16 偏移 | 纹理解码硬件＋融合运算 | 少搬权重，少占 shader 解压算力 | 有损，且只证明该模型路径 |

## 6. 对移动神经图形的实际意义

**小 CNN 不一定像大语言模型那样受权重带宽主导。** 卷积权重在很多像素上复用，若已留在缓存，继续压缩可能主要改善缓存占用和能耗；全分辨率 activation、历史帧、光流及前后处理反而可能占主要流量。这与 [NSS 的共享带宽设计约束][a9]和 [Arm 对 FC／卷积访存差异的说明][a4]一致。

可用一个简单模型判断收益上限。设原总流量中权重占比为 `f`，权重流量压缩到原来的 `1/R`，其他流量不变、忽略元数据，则总流量减少比例为：

`f × (1 − 1/R)`

例如权重只占 20%，压缩 2 倍只减少 10% 总流量；权重占 80% 时才减少 40%。这仍是流量收益，不是延迟收益：还要看算力、解码速度、同步开销及是否压在关键路径上。

因此评估 NX compressed weights，最值得补齐的是以下四组数据：

| 要验证的事情 | 应记录的证据 |
| --- | --- |
| 真正压缩了多少 | 逻辑权重字节、编译后编码字节、元数据／对齐、运行时额外副本 |
| 压缩状态保持到哪里 | 冷／热缓存的 DRAM 流量、L2 命中与流量、片上存储占用 |
| 解码能否跟上 | 解码输出吞吐、MAC 等待周期、与图形并发时的吞吐和功耗 |
| 最终是否值得 | 固定模型、精度、频率与画质基线的逐层时间、整帧时间和能量 |

优先比较同一量化模型的压缩开关；若驱动不暴露开关，应明确承认无法隔离单项收益。不要把 FP32→INT8、渲染分辨率降低、插帧和权重编码的综合收益都记到 compressed weights 名下。

**架构上的判断：** 这不是一个脱离模型和调度的“小解压模块”。它是模型表示、编译布局、内存层次与执行单元共同制定的供数方案。Arm NX 的产品方向已经确认；目前最需要等待的是格式与后端细节，而不是再猜一个压缩算法名称。

## 7. 一手资料与源码索引

下面按推荐阅读顺序列出来源。日期指发布年或明确的软件版本；源码快照均核查于 2026-09-09。动态文档与产品页可能更新。

| 来源 | 直接入口 | 用于核实 |
| --- | --- | --- |
| Arm，2026-09-08 | [NX 发布文章][a1]、[原始架构图][a2] | compressed weights 原始声明 |
| Arm / Ian Bratt，Hot Chips 30，2018 | [First-Generation ML Processor，p13、17–20][a3] | 压缩 SRAM → decoder → MAC 的历史路线 |
| Arm / George Gekov，2023 | [Pruning and Clustering for Ethos-U][a4] | 有损模型优化与无损编码的边界 |
| Arm Vela，`a43300cb989c` | [权重重排][a5]、[编码][a6]、[解码][a7] | 可检查的真实 codec |
| Arm，2024 | [Ethos-U85 发布][a8] | decoder 与 2:4 计算是不同能力 |
| Arm，2025／2026 | [NSS 技术演讲][a9]、[NFRU 技术演讲][a10] | 算法、精度与带宽的共同设计 |
| Arm Model Gym，`b86ee99125ea` | [导出流程][a11]、[NSS sparse filter][a16] | INT8 导出及避免误读 sparse 名称 |
| Arm Hugging Face | [NSS，`a8ed73709fc8`][a12]、[NFRU，`f6ae2bd4b019`][a13] | 检查点、VGF、元数据的发布形态 |
| Arm VGF／Emulation | [VGF，`c465f987378b`][a14]、[Emulation，`c604755e512a`][a15] | 软件公开边界 |
| Khronos | [Graph Constants 示例][a17]、[常量 API][a18]、[稀疏提示][a19] | 应用接口与硬件实现分离 |
| Qualcomm，2023 | [Snapdragon 8 Gen 2 AI deep dive][q1] | INT4、micro-tile、direct link |
| Qualcomm AIMET 2.19／源码 `89ab3cb197be` | [LPBQ 文档][q2]、[量化器源码][q3] | 存储位宽与还原位宽分离 |
| Qualcomm AI Hub | [Qwen3 导出][q4]、[模型卡][q5]、[精度定义][q6] | 可部署资产与 W4A16 语义 |
| Qualcomm，2024-01 | [Cloud AI 100 MXFP6][q7] | 软件流式解压后 FP16 运算 |
| NVIDIA NVDLA | [压缩格式][n1]、[解码 RTL][n2] | 位掩码、分组大小、解码通路 |
| NVIDIA，2020／2021 | [Ampere 架构][n3]、[cuSPARSELt 示例][n4] | 结构化稀疏直接减少计算 |
| NVIDIA，2024／2025 | [TensorRT WoQ][n5]、[NVFP4][n6]、[CUTLASS][n7] | 压缩存储与低精度算术区别 |
| NVIDIA，2026 | [Rubin 架构][n8] | 3 位 LUT 权重与内联查表 |
| Apple，WWDC23 | [Core ML 模型压缩][p1] | 从提前展开到即时解压 |
| Apple Core ML Tools，源码 `c59d1a2fe535` | [部署指南][p2]、[MIL 算子][p3]、[Stable Diffusion][p4]、[palette 说明][p5] | 码表、反量化及后端差异 |
| Apple，2025 | [AFM 技术报告 §6][p6]、[官方研究入口][p7] | ASTC 权重及偏移开销 |

补充论文入口：[Deep Compression，Han 等，2015][r1]解释剪枝、权重共享与编码组合；[EIE，Han 等，2016][r2]研究直接消费压缩模型的专用加速器。它们是理解技术背景的原始论文，**不是 Arm NX 实现证明**。Arm 2018 演讲亦引用 Han 等的剪枝研究。

[a1]: https://newsroom.arm.com/blog/arm-mali-g2-ultra-nx-ai-native-mobile-graphics
[a2]: https://newsroom.arm.com/wp-content/uploads/2026/09/Arm-Newsroom-GPU-Tech-Day-Final-slide-23-1200x675.png
[a3]: https://old.hotchips.org/hc30/2conf/2.07_ARM_ML_Processor_HC30_ARM_2018_08_17.pdf#page=19
[a4]: https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/pruning-clustering-arm-ethos-u-npu
[a5]: https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/a43300cb989cd7115387798162115a586802db3e/ethosu/vela/weight_compressor.py#L141
[a6]: https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/a43300cb989cd7115387798162115a586802db3e/ethosu/mlw_codec/mlw_encode.c#L560
[a7]: https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/a43300cb989cd7115387798162115a586802db3e/ethosu/mlw_codec/mlw_decode.c#L270
[a8]: https://newsroom.arm.com/blog/ethos-u85
[a9]: https://huggingface.co/Arm/neural-super-sampling/resolve/a8ed73709fc821f2b50c35455aeb708b9a89c096/2025-neural-super-sampling.pdf#page=19
[a10]: https://huggingface.co/Arm/neural-frame-rate-upscaling/resolve/f6ae2bd4b019f5f57a965ce0ad53bccba9a4f6fd/2026-neural-frame-rate-upscaling.pdf#page=74
[a11]: https://github.com/arm/neural-graphics-model-gym/blob/b86ee99125ea01c9ec1acf471743e5eb2478414a/src/ng_model_gym/core/export/model_export.py#L183
[a12]: https://huggingface.co/Arm/neural-super-sampling/tree/a8ed73709fc821f2b50c35455aeb708b9a89c096
[a13]: https://huggingface.co/Arm/neural-frame-rate-upscaling/tree/f6ae2bd4b019f5f57a965ce0ad53bccba9a4f6fd
[a14]: https://github.com/arm/ai-ml-sdk-vgf-library/blob/c465f987378b91de196edec05828f951c44ee1d7/src/encoder.cpp#L304
[a15]: https://github.com/arm/ai-ml-emulation-layer-for-vulkan/blob/c604755e512a14a94b05e2ac33f148b1a5f027b0/docs/source/data_graph.rst
[a16]: https://github.com/arm/neural-graphics-model-gym/blob/b86ee99125ea01c9ec1acf471743e5eb2478414a/src/ng_model_gym/usecases/nss/model/torch_postprocess/filter.py#L21
[a17]: https://docs.vulkan.org/samples/latest/samples/extensions/tensor_and_data_graph/graph_constants/README.html
[a18]: https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphPipelineConstantARM.html
[a19]: https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM.html
[q1]: https://www.qualcomm.com/news/onq/2023/03/snapdragon-8-gen-2-ai-powerhouse-deep-dive-video
[q2]: https://qualcomm.github.io/aimet-pages/releases/2.19.0/techniques/lpbq.html
[q3]: https://github.com/qualcomm/aimet/blob/89ab3cb197be344e6f623366556198aa1a78e435/TrainingExtensions/torch/src/python/aimet_torch/quantization/affine/quantizer.py#L1231
[q4]: https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/qwen3_4b/README.md
[q5]: https://huggingface.co/qualcomm/Qwen3-4B
[q6]: https://workbench.aihub.qualcomm.com/docs/hub/api.html
[q7]: https://www.qualcomm.com/developer/blog/2024/01/qualcomm-cloud-ai-100-accelerates-large-language-model-inference-2x-using-microscaling-mx
[n1]: https://nvdla.org/hw/format.html#sparse-compression-option
[n2]: https://github.com/nvdla/hw/blob/master/vmod/nvdla/csc/NV_NVDLA_CSC_WL_dec.v
[n3]: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
[n4]: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSELt/matmul/matmul_example.cpp
[n5]: https://developer.nvidia.com/blog/nvidia-tensorrt-10-0-upgrades-usability-performance-and-ai-model-support/
[n6]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
[n7]: https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu
[n8]: https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/
[p1]: https://developer.apple.com/videos/play/wwdc2023/10047/
[p2]: https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
[p3]: https://github.com/apple/coremltools/blob/c59d1a2fe535367db7b9b95a8cc2cffaa82dac4d/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py#L24
[p4]: https://github.com/apple/ml-stable-diffusion#compression-lower-than-6-bits
[p5]: https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html
[p6]: https://arxiv.org/html/2507.13575v1#S6
[p7]: https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025
[r1]: https://arxiv.org/abs/1510.00149
[r2]: https://arxiv.org/abs/1602.01528

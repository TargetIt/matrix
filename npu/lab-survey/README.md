# 神经网络、Transformer 与 AI 加速器 Lab 调研

调研日期：2026-08-31

## 结论

如果目标是从“会训练模型”逐步走到“理解和实现 NPU”，不建议一开始购买开发板。更有效的顺序是：

1. 用 Colab/Jupyter 补齐反向传播、CNN 和 Transformer。
2. 在电脑上完成量化、剪枝、算子优化和模型部署实验。
3. 用 SCALE-Sim、Timeloop/Accelergy 研究 systolic array、dataflow、存储层次和能耗。
4. 用 hls4ml、FINN、CFU-Playground 或 Gemmini 进入 HLS、编译器、RTL 和软硬件协同。
5. 最后再根据方向购买 Raspberry Pi、Arduino/MCU、Jetson 或 FPGA 板。

对当前电脑（Apple M4 Pro、macOS arm64），最顺畅的起点是 Karpathy Zero to Hero、CS231n、MIT 6.5940 的公开 Colab、Stanford CS336 Assignment 1、MLC.ai 和 SCALE-Sim。FINN、Timeloop、Gemmini 与 FPGA 综合工具更适合 Ubuntu x86_64 主机、虚拟机或云服务器。

## 筛选标准

本报告优先收录满足以下条件的项目：

- 有公开、可复现的代码或 Notebook，而不只是讲义和视频。
- 有明确实验目标、步骤、测试或可观察的性能结果。
- 来源是大学课程、研究团队或工具官方组织。
- 截至调研日仍可访问；维护状态和历史版本会明确标注。
- 区分“无需开发板”“完整实验需要 GPU”和“必须连接真实硬件”。

“电脑可运行”不等于“任何电脑都能快速完成”：大模型训练、RTL 仿真和设计空间搜索可能需要数小时，某些 Docker 镜像及 FPGA 工具只支持 x86_64 Linux。

## 第一组：无需开发板

### 总表

| 优先级 | Lab | 主要内容 | 最低运行方式 | 完整实验资源 | 难度 |
| ---: | --- | --- | --- | --- | --- |
| 1 | [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | autograd、MLP、BatchNorm、CNN、GPT | 浏览器 Colab 或本地 Python | CPU 可学习，小模型训练建议 GPU/MPS | 入门 |
| 2 | [Stanford CS231n 2025 Assignments](https://cs231n.github.io/assignments2025/assignment1/) | NumPy 神经网络、反向传播、CNN、PyTorch、RNN | 官方 Colab | 免费 Colab 足够 | 入门—中级 |
| 3 | [MIT 6.5940 Fall 2024 Labs](https://hanlab.mit.edu/courses/2024-fall-65940) | 剪枝、量化、NAS、LLM 压缩与笔记本部署 | 5 个公开 Colab | 部分实验建议 Colab GPU | 中级 |
| 4 | [Stanford CS336 Assignments](https://cs336.stanford.edu/) | tokenizer、Transformer、训练、系统优化、数据与对齐 | Assignment 1 可本地测试 | 完整训练/系统实验需要 NVIDIA GPU 或云资源 | 中高级 |
| 5 | [MLC.ai Notebooks](https://github.com/mlc-ai/notebooks) | TensorIR、算子调度、自动优化、GPU/tensor core、计算图 | Jupyter；CPU 章节可直接跑 | GPU 专章需要 CUDA/兼容环境 | 中高级 |
| 6 | [SCALE-Sim v3](https://github.com/scalesim-project/SCALE-Sim) | systolic array、GEMM/卷积/Attention、多核、带宽、能耗 | Python + CSV 配置 | 纯 CPU；大规模 sweep 耗时增加 | 中级 |
| 7 | [Timeloop/Accelergy Exercises](https://github.com/Accelergy-Project/timeloop-accelergy-exercises) | dataflow、mapping、存储层次、能耗、稀疏加速器 | Docker + Jupyter | 推荐 Ubuntu x86_64 | 高级 |
| 8 | [hls4ml Tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial) | Keras/PyTorch 到 HLS、量化、reuse factor、综合报告 | 训练/转换 Notebook 可先跑 | 综合需 AMD Vitis HLS，通常为 x86 Linux | 中高级 |
| 9 | [FINN Tutorials](https://github.com/Xilinx/finn/tree/main/notebooks) | 低比特 QNN、ONNX 变换、dataflow accelerator build | 官方 Docker | 软件变换无需板；综合阶段需要 Vivado/较大内存 | 高级 |
| 10 | [CFU-Playground Simulator](https://github.com/google/CFU-Playground) | 为 RISC-V 软核添加 ML 自定义指令，优化 TFLM kernel | Renode 或 Verilator | Linux 环境；Renode 快、Verilator 更精确但慢 | 高级 |
| 11 | [Gemmini](https://github.com/ucb-bar/gemmini) | systolic array、RoCC、自定义 ISA、scratchpad/DMA、软硬件栈 | Spike 或 Verilator | Ubuntu x86_64、Chipyard；编译和仿真较重 | 高级 |

### 1. Karpathy — Neural Networks: Zero to Hero

推荐作为第一套实验。它不是只调用框架 API，而是从标量 autograd 和反向传播开始，逐步实现字符语言模型、MLP、BatchNorm、手写反向传播、WaveNet 风格网络，最后构造 GPT。Notebook、练习和视频互相对应。

适合完成的产出：

- 独立实现一个 micrograd，并用梯度检查验证。
- 比较手写反向传播与 PyTorch autograd。
- 在小语料上训练字符级 GPT，记录 context length、embedding size、head 数和损失的关系。
- 为后续加速器学习整理出 Transformer 的核心算子：embedding、GEMM、attention、normalization 和激活函数。

局限：它主要建立模型直觉，不涉及硬件性能建模；仓库最近代码更新为 2024，但内容仍是很好的基础。

### 2. Stanford CS231n 2025 Assignments

Assignment 1 用 NumPy 实现 Softmax、两层网络和全连接网络；Assignment 2 覆盖 BatchNorm、LayerNorm、Dropout、卷积、PyTorch、RNN 图像描述。官方工作流使用 Colab，适合在购买硬件前建立张量形状、梯度和算子成本意识。

建议至少完成：

- A1 的 Two-Layer Neural Network 与 FullyConnectedNets。
- A2 的 ConvolutionalNetworks 与 PyTorch on CIFAR-10。
- 给卷积和全连接层增加 MAC 数、参数量、激活量统计，为后续 NPU 实验建立工作负载表。

课程页说明本地环境不是官方主要支持方式，因此当前 Mac 上优先使用 Colab。

### 3. MIT 6.5940 — TinyML and Efficient Deep Learning Computing

Fall 2024 课程页公开了 5 个 Colab Lab，主题覆盖剪枝、量化、神经架构搜索、LLM 压缩和在笔记本电脑上部署 LLM。这是从“模型学习”进入“高效 AI/NPU”最短的一座桥。

公开入口：

- [Lab 0](https://colab.research.google.com/drive/1gvxq7mIAeIBAtmLKH1Q1GknA-GRsK7Q6)
- [Lab 1](https://colab.research.google.com/drive/1Fagq3JQBzCizodyxpHKvWDzfCC7F1RWN)
- [Lab 2](https://colab.research.google.com/drive/11IBla1q1McoZ2oCANCGHns8VtzG5nCMP)
- [Lab 3](https://colab.research.google.com/drive/1xKReLBHVS6bkFbYkfi-Ky3C4loQmG6Yc)
- [Lab 4](https://colab.research.google.com/drive/16H9RvSg4XIF35X3fLGQUVwAE9ccvDj14)

MIT Fall 2026 页面已经列出新一轮 Lab 计划，但截至调研日链接尚未全部公开，因此当前应使用可直接访问的 2024 Colab，而不是等待未发布材料。

### 4. Stanford CS336 — Language Modeling from Scratch

这是进阶 Transformer 主线。2026 公开仓库包括：

- [Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)：tokenizer、Transformer LM、优化器、训练与测试；使用 `uv`，可在电脑上先跑单元测试和小数据。
- [Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems)：优化 Transformer、分布式训练与系统性能。
- [Assignment 3: Scaling](https://github.com/stanford-cs336/assignment3-scaling)：训练运行与 scaling law 实验。
- [Assignment 4: Data](https://github.com/stanford-cs336/assignment4-data)：数据过滤、去重和训练；课程完整配置使用 8 张 B200，不属于普通电脑任务。
- [Assignment 5: Alignment](https://github.com/stanford-cs336/assignment5-alignment)：GRPO/对齐实验。

不要把五个作业都列为“Mac 可直接完成”。推荐当前机器只从 Assignment 1 的 tokenizer、模型实现、单元测试和 TinyStories 小规模训练开始；Assignment 2 以后按需使用云端 NVIDIA GPU。

### 5. MLC.ai — Machine Learning Compilation

官方 Notebook 依次覆盖 Tensor Program 抽象、TensorIR、端到端模型、自动程序优化、框架集成、GPU/专用硬件和计算图优化，并包含 tensor core 教程。它回答的是“框架中的算子怎样被降级、调度并映射到硬件”，比一般模型部署教程更接近 NPU 编译器。

推荐实验：

- 修改矩阵乘的 tiling、reorder、vectorize，比较运行时间。
- 对同一算子比较朴素实现、自动调度和手工调度。
- 学完 GPU 章节后，把概念映射到 NPU：线程层次对应 PE 阵列，shared/local memory 对应片上 buffer/register。

CPU 章节可在当前 Mac 上尝试；CUDA/tensor core 章节应转到 NVIDIA 环境。

### 6. SCALE-Sim v3

这是最适合第一步学习 AI 加速器的纯软件模拟器。它用 Python 模拟 systolic array，支持卷积、全连接及 Attention 等 GEMM 工作负载，并可改变阵列大小、dataflow、SRAM、带宽、多核配置和稀疏性。

建议设计一个小型实验矩阵：

1. 固定 GEMM，比较 8×8、16×16、32×32 阵列。
2. 比较 output-stationary、weight-stationary 与 input-stationary。
3. 扫描 SRAM 和片外带宽，找出 compute-bound 与 memory-bound 的分界。
4. 输入 Transformer 的 Q/K/V、attention score 和 FFN GEMM，比较长序列下利用率。

仓库在 2026 年仍更新，是本报告中最推荐的“电脑直接运行的 NPU Lab”。

### 7. Timeloop / Accelergy Exercises

这套官方练习用 Docker 启动 Jupyter，研究 mapping、数据复用、层级存储访问和能耗。示例包括 Eyeriss-like、Simba-like、output/weight-stationary、sparse tensor core，并已包含 Vision Transformer layer shapes。

它比 SCALE-Sim 更难，但能回答更完整的问题：同一个张量运算放进不同空间/时间 mapping 后，DRAM、SRAM、NoC 和 MAC 分别消耗多少能量。建议在完成 SCALE-Sim 后进入。

当前电脑为 arm64；官方 Docker/依赖的最稳妥环境仍是 Ubuntu x86_64。可用云 VM 或独立 Linux 主机，避免先在 macOS 上处理交叉架构兼容问题。

### 8. hls4ml Tutorial

教程 Notebook 从 Keras/PyTorch 训练开始，继续到 HLS 转换与综合，并覆盖 QKeras、Brevitas、reuse factor、profiling、量化 CNN 和 SNN。无需开发板也能完成模型训练、量化和部分转换；要得到真实综合后的 latency/resource 报告，需要安装 AMD Vitis HLS。

它适合观察以下关系：

- 位宽降低如何改变 DSP/LUT/BRAM。
- reuse factor 如何在并行度、吞吐和资源之间取舍。
- 模型精度与硬件成本如何共同优化。

Binder 可打开 Notebook，但综合工具不会由 Binder 提供。Apple Silicon 不是 Vitis HLS 的标准平台，建议使用 x86_64 Linux。

### 9. FINN Tutorials

FINN 专门面向低比特量化神经网络的 FPGA dataflow accelerator。官方只支持 Docker 执行编译器，并提供从 Brevitas/QONNX 到 dataflow partition、streamlining、折叠参数和硬件生成的 Jupyter 教程。

无需板卡时可以学习 ONNX 图变换、量化数据类型、每层 PE/SIMD 折叠和吞吐估计。只有在实际部署或验证 bitstream 时才需要 FPGA。完整 synthesis 通常还需要 Vivado，因此本地软件阶段与硬件阶段要分开规划。

### 10. CFU-Playground：先仿真、后上板

CFU-Playground 的教学目标很清楚：给 RISC-V 软核增加 Custom Function Unit，修改 TFLite Micro kernel 使用新指令，再比较每层 cycle count。它是少见的可以先用 Renode/Verilator、之后用同一工程上 FPGA 的连续 Lab。

- 无开发板：`make renode` 做快速功能仿真；Verilator 做较慢的 cycle-accurate RTL 仿真。
- 有开发板：在 LiteX 支持板上测量真实运行；官方列出 Arty A7-35T/100T、iCEBreaker、Fomu、OrangeCrab、ULX3S、Nexys Video 等已测试目标。

项目 README 也提醒这是探索型框架，文档和构建偶尔可能破损。它适合作为高级项目，不应放在入门第一周。

### 11. Gemmini

Gemmini 是 Berkeley Chipyard 生态中的完整 DNN accelerator：Chisel RTL、RoCC 自定义指令、systolic array、scratchpad、accumulator、DMA、软件库和 DNN workload 全部开放。

无需 FPGA 时可先用 Spike 验证功能，再用 Verilator 做 cycle-accurate 仿真；可修改阵列尺寸、dataflow、数据类型、队列和存储参数。它最接近“设计一个可集成进 SoC 的 NPU”，但依赖多、编译重、仿真慢，建议在 SCALE-Sim、Timeloop 和 CFU 之后学习。

## 第二组：需要开发板或专用硬件

### 总表

| 优先级 | Lab | 推荐硬件 | 学习重点 | 说明 |
| ---: | --- | --- | --- | --- |
| 1 | [Stanford EE292D Labs](https://github.com/ee292d/labs) | Raspberry Pi 5 | 本地 LLM、视觉、语音、翻译、微调 | 最接近通用端侧 AI 产品实践 |
| 2 | [Harvard TinyMLx](https://github.com/tinyMLx/courseware) | Arduino Tiny Machine Learning Kit / Nano 33 BLE Sense + Shield | TFLite Micro、传感器、KWS、手势、相机、量化 | MCU 资源约束最直观 |
| 3 | [Jetson Hello AI World](https://github.com/dusty-nv/jetson-inference) | Jetson Orin Nano 优先 | TensorRT、摄像头、分类/检测/分割、端侧训练 | GPU 边缘计算；后续可做 ViT/LLM/VLM |
| 4 | [FINN Examples](https://github.com/Xilinx/finn-examples) | Pynq-Z1、Ultra96、ZCU104；或 Alveo U250 | 低比特 QNN dataflow accelerator 与真实吞吐 | 预构建 bitstream，入门比从零综合容易 |
| 5 | [CFU-Playground](https://github.com/google/CFU-Playground) | Arty A7-35T/100T 等 LiteX 板 | 自定义指令、TFLM kernel、cycle 对比 | 先仿真后上板，软硬件协同最完整 |
| 6 | [TFLite Micro Examples](https://github.com/tensorflow/tflite-micro) + [MLPerf Tiny](https://github.com/mlcommons/tiny) | Arduino、ESP、Coral Dev Board Micro 或厂商 MCU/DSP | 模型移植、内存占用、标准 benchmark | 更像参考工程/基准套件，不是连续课程 |

### 1. Raspberry Pi 5：继续 EE292D

已经发现的 EE292D 是最适合第一块板的课程：Lab 1–6 覆盖本地 LLM、图像分类、目标/人体定位、Whisper、翻译和 LLM 微调。它主要训练 Linux 边缘设备上的模型部署与优化，不会深入自定义 NPU RTL。

推荐扩展作业：对每个模型记录 CPU 占用、峰值内存、模型大小、首 token 延迟、tokens/s 和功耗；再比较 FP32、FP16、INT8 或不同推理后端。

### 2. Arduino / MCU：Harvard TinyMLx

TinyMLx 提供课程 syllabus、assignments/labs、代码 walkthrough、Colab 和 Arduino library。公开 Colab 从第一个神经网络、卷积、Fashion-MNIST，延伸到 TFLite Converter、PTQ/QAT、关键词识别、autoencoder、摄像头和自定义手势。

学习时应分两阶段：

- 先在 [tinyMLx/colabs](https://github.com/tinyMLx/colabs) 完成训练、量化和模型转换，不需要开发板。
- 再用 [tinyMLx/arduino-library](https://github.com/tinyMLx/arduino-library) 部署到 TinyML Kit/Shield，观察 SRAM、Flash、采样和实时性限制。

购买前需核实课程所用 Shield、摄像头和 Arduino 板版本的兼容性，不要只按“Nano 33”名称购买。

### 3. Jetson：Hello AI World 与 Jetson AI Lab

`jetson-inference` 是结构化 instructional guide，包含 TensorRT 推理、PyTorch transfer learning、摄像头数据采集、分类、检测、分割和姿态估计。若重点转向 Transformer，可继续 Jetson AI Lab 的 ViT、VLM 和本地 LLM 教程。

新购硬件优先考虑 Jetson Orin Nano，而不是较老的 Jetson Nano；仓库明确支持 Orin/JetPack 6。Jetson 学到的是 GPU/TensorRT 边缘栈，不是独立 NPU 微架构。

### 4. FPGA：FINN Examples

官方提供预构建 bitstream、PYNQ Python driver 和 Jupyter Notebook，示例包括 MNIST MLP、CIFAR-10 CNV、语音命令和网络入侵检测等低比特模型。

已正式提供 bitstream 的边缘板主要是 Pynq-Z1、Ultra96 和 ZCU104；README 特别说明 Pynq-Z2 可手工适配，但不是预构建示例的正式测试目标。若只是学习，不应因为 Pynq-Z2 更常见就假设所有 Notebook 可直接运行。

### 5. FPGA 自定义指令：CFU-Playground

推荐先完成仿真版，再购买 Arty A7。上板后的核心实验是比较修改前后的 TFLM layer cycles，而不是只证明 bitstream 能启动。Arty A7-35T/100T 文档和社区经验相对多，是比稀有 LiteX 板更稳妥的选择。

### 6. MCU 基准：TFLite Micro 与 MLPerf Tiny

TFLite Micro 官方仓库列出 Arduino、Coral Dev Board Micro、Espressif、Renesas、Silicon Labs、TI 等社区支持示例，并带 Renode 软件仿真说明。MLPerf Tiny 提供代表性的 image classification、keyword spotting、visual wake words 和 anomaly detection benchmark。

它们适合完成基础课程后的“跨板比较”：统一模型和数据集，记录 latency、energy、Flash、SRAM 与 accuracy。但它们不是像 EE292D 那样按周组织的课程，初学者应先完成 TinyMLx。

## 针对当前电脑的推荐路线

### 第 1–4 周：模型基础，不买板

1. Zero to Hero：micrograd、makemore、GPT。
2. CS231n：两层网络、全连接、卷积、PyTorch。
3. 每个网络统一统计参数量、MAC、activation bytes 和训练/推理时间。

### 第 5–7 周：Transformer 与效率

1. CS336 Assignment 1：tokenizer、Transformer、AdamW 和小数据训练。
2. MIT 6.5940：剪枝、量化与 LLM 压缩 Colab。
3. 在 M4 Pro 上可尝试 PyTorch MPS；若课程代码只支持 CUDA，则直接使用 Colab，不花时间改底层依赖。

### 第 8–10 周：加速器建模

1. SCALE-Sim：阵列大小、dataflow、带宽、SRAM sweep。
2. MLC.ai：矩阵乘 tiling 和 schedule。
3. Timeloop/Accelergy：迁移到 Ubuntu x86_64，分析 Eyeriss-like 与 Transformer layer shapes 的能耗。

### 第 11–12 周：选择方向

- 端侧产品：购买 Raspberry Pi 5，继续 EE292D。
- 极低功耗 MCU：购买兼容的 Arduino TinyML kit，继续 TinyMLx。
- GPU 边缘与 Transformer：选择 Jetson Orin Nano。
- FPGA dataflow：先跑 FINN Docker，再根据正式支持清单选择板。
- 自定义 NPU 指令/RTL：先跑 CFU Renode/Verilator 或 Gemmini，再考虑 Arty A7/更高端 FPGA。

## 不建议当前作为主线的材料

- [Xilinx DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ)：仓库已归档，适合查旧工程，不适合作为新学习路线。
- [pulp-platform/pulp-training](https://github.com/pulp-platform/pulp-training)：仓库已归档且更新停在 2023，除非明确研究 PULP 历史工具链。
- TVM VTA 老教程：当前 Apache TVM `main` 已不再保留旧 VTA 目录，网络上的 PYNQ/DE10-Nano 教程容易与现行 TVM 不兼容。
- Cornell ECE5545 的若干学生作业镜像：内容有参考价值，但不是官方统一维护的 Lab 发布源，复现性弱于本报告的主清单。
- MIT 6.5940 Fall 2026 Labs：课程计划已发布，但调研日公开链接尚不完整；先用 2024 的五个 Colab。

## 维护状态与来源核验

下表的“最近更新”取自 GitHub API 的 `pushed_at`，只表示仓库活动，不能单独代表内容质量。

| 项目 | 最近更新 | 许可证/状态 |
| --- | --- | --- |
| Stanford CS336 Assignment 1 | 2026-04-07 | MIT，活跃 |
| SCALE-Sim | 2026-06-28 | MIT，活跃 |
| hls4ml Tutorial | 2026-08-17 | 仓库未声明标准 SPDX；活跃 |
| FINN | 2026-08-31 | BSD-3-Clause，活跃 |
| Gemmini | 2026-06-30 | 仓库 API 未识别 SPDX；活跃 |
| CFU-Playground | 2026-02-26 | Apache-2.0，活跃 |
| Timeloop/Accelergy Exercises | 2025-04-09 | MIT，活跃 |
| MLC.ai Notebooks | 2024-11-22 | Apache-2.0，可用但更新较慢 |
| Zero to Hero | 2024-08-18 | MIT，内容稳定 |
| EE292D Labs | 2025-05-06 | Apache-2.0 |
| tinyMLx Colabs | 2026-06-27 | 仓库 API 未识别 SPDX；持续更新 |
| TFLite Micro | 2026-08-28 | Apache-2.0，活跃 |
| MLPerf Tiny | 2026-07-13 | Apache-2.0，活跃 |
| FINN Examples | 2026-03-10 | BSD-3-Clause，活跃 |
| Jetson Inference | 2025-10-16 | MIT，活跃 |

## 最终优先清单

如果只选六套，建议按以下顺序：

1. Zero to Hero：真正理解神经网络和 GPT。
2. MIT 6.5940 Colabs：模型压缩和高效 AI。
3. Stanford CS336 Assignment 1：从 tokenizer 到 Transformer 训练。
4. SCALE-Sim：第一套 NPU/systolic array 实验。
5. Timeloop/Accelergy：dataflow、存储和能耗建模。
6. CFU-Playground：从仿真过渡到 FPGA 自定义加速单元。

购买开发板前，先完成前五项中的至少三项。这样才能根据真正感兴趣的是部署、编译器、架构还是 RTL 来选板，而不是先买硬件再寻找用途。

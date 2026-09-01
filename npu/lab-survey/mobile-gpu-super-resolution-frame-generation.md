# 移动 GPU 实时超分辨率与插帧实验调研

调研日期：2026-08-31

## 结论

如果目标是为自研移动 GPU 评估类似 DLSS、FSR、XeSS 的“超分 + 插帧”能力，最有效的起点不是直接复刻 DLSS，也不是先购买 FPGA，而是建立一条能够在同一批渲染数据上比较算法、GPU 路径和硬件代价的实验链。

建议优先顺序如下：

1. 先跑 [Mob-FGSR](https://github.com/Mob-FGSR/MobFGSR)，它是当前最贴合“移动 GPU + 超分 + 插帧”目标的公开研究代码。
2. 在 Android Vulkan 上跑 [Snapdragon GSR 2](https://github.com/SnapdragonGameStudios/snapdragon-gsr) 及其[官方 Vulkan 样例](https://github.com/SnapdragonGameStudios/adreno-gpu-vulkan-code-sample-framework/tree/main/samples/sgsr2)，建立纯 shader 时域超分基线。
3. 跑 [Arm ASR](https://github.com/arm/accuracy-super-resolution) 与 [Arm Neural Graphics SDK](https://github.com/arm/neural-graphics-sdk-for-game-engines)，分别研究移动优化的非神经时域超分，以及 NSS/NFRU 神经超分和插帧。
4. 在 Windows 独显上跑 [AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) 的 FSR 3.1 路径，理解完整的 optical flow、frame interpolation、swapchain、UI 和 pacing 设计。
5. 最后把同一批 color、depth、motion vector、exposure 和 UI 数据送入自研移动 GPU，比较 fragment、compute、专用光流单元和神经网络单元的成本。

最值得直接借鉴的并非某个厂商的模型权重，而是公共接口和数据流：低分辨率颜色、深度、运动矢量、抖动、曝光、reactive mask、无 UI 颜色、光流、历史缓冲、插值帧以及显示队列之间如何连接。

## 先把四类技术分开

| 类型 | 典型方案 | 输入 | 优点 | 主要问题 |
| --- | --- | --- | --- | --- |
| 空间超分 | FSR 1、SGSR 1 | 当前低分辨率颜色 | 简单、单帧、低延迟 | 不能恢复时间信息，细线和运动画质有限 |
| 时域超分 | FSR 2/3、SGSR 2、Arm ASR、MetalFX Temporal | 颜色、深度、motion vector、history、jitter | 画质明显好于纯空间方法 | history rejection、遮挡、透明物和带宽复杂 |
| 帧插值 | FSR 3 FG、DLSS-FG、XeSS-FG、Arm NFRU | 前后已渲染帧、MV、深度，常叠加 optical flow | 画质通常比外推稳定 | 必须等待后一帧，至少引入一个渲染帧级的额外延迟 |
| 帧外推 | Mob-FGSR-E、ExtraNet、Qualcomm AFME | 当前及历史帧、运动信息或姿态 | 可降低交互延迟，适合 VR/低延迟 | 新暴露区域没有未来信息，伪影风险更高 |

这四类方案不能只用“显示 FPS”横向比较。插值可能把 30 个真实渲染帧显示成 60 帧，但输入采样、游戏逻辑和真实渲染仍是 30 Hz；若没有低延迟和正确的 frame pacing，数字变大不等于操作更流畅。

## 最值得做的公开实验

### 1. [Mob-FGSR：移动超分与插帧联合实验](https://github.com/Mob-FGSR/MobFGSR)

直达：[项目页](https://mob-fgsr.github.io/) · [论文](https://mob-fgsr.github.io/resources/paper/SIGGRAPH_Conf_Mob_FGSR.pdf) · [源码](https://github.com/Mob-FGSR/MobFGSR)

**核心算法**

- 输入相邻真实渲染帧的 color、depth 和 motion vector；插值路径先做 depth-aware motion splatting，以深度竞争处理前后景，再得到中间时刻的双向运动。
- 外推路径只使用当前帧和历史帧预测未来运动，通过运动传播与空洞填补处理未来时刻新暴露的区域；随后统一执行 warp、遮挡修复和 blend。
- 超分路径采用时域累积，并用学习得到的 LUT 保存 16 像素邻域重采样权重；权重推理已离线折叠进查找表，运行时没有神经网络、TensorRT 或 NPU 依赖。

**复现平台要求**

- 官方程序：Linux/Windows 上支持 OpenGL 4.3 compute shader 的 GPU，C++11 编译器和 CMake；准备连续帧的 color/depth/motion-vector 数据，并按仓库说明配置输入目录。
- 论文移动结果：Snapdragon 8 Gen 3/Adreno；公开仓库并不是 Android 游戏插件，因此要在移动 GPU 上复现需自行移植 EGL/Android 或 Vulkan，并接入引擎 G-buffer。
- 当前 M4 Pro 的原生 OpenGL 只到 4.1，不能原样构建 4.3 compute 路径；可用于代码分析和 Vulkan/Metal 移植，但不能作为官方代码的直接运行平台。

这是本调研的第一优先级。项目来自浙江大学、OPPO 计算与图形研究院和 UCSB，发表于 SIGGRAPH 2024，目标就是在缺少桌面级 optical-flow 硬件的移动 GPU 上同时完成 frame generation 与 super resolution。

它的特点是：

- 只依赖渲染器可提供的 color、depth 和 motion vector。
- 同时提供插值、外推、超分、插值 + 超分、外推 + 超分几种模式。
- 使用 motion splatting、深度优先、遮挡填补和 LUT 重采样，不依赖神经网络运行时。
- 官方代码是 C++、OpenGL 4.3 和 compute shader，适合拆分成逐 pass 微基准。
- 论文在 Snapdragon 8 Gen 3 上报告：720p 插值约 2.2 ms、外推约 1.9 ms；从 540p 到 1080p 的联合模式约 1.8–2.3 ms。

建议实验：

1. 先使用仓库数据跑通五种模式，生成每个 pass 的中间图。
2. 将 motion splat、空洞填补、warp、blend 和 LUT SR 分开计时。
3. 对 atomic splat 改写局部 tile/binning 版本，比较原子冲突、带宽和片上存储需求。
4. 分别测试插值与外推；不要把两者合成一个“插帧”指标。
5. 将 OpenGL compute shader 移植到 Vulkan，作为自研 GPU 的第一套跨平台 workload。

限制：公开仓库更像论文复现程序，不是完整游戏引擎插件；当前 Mac 的原生 OpenGL 最高仅 4.1，不能直接满足其 OpenGL 4.3 要求，优先使用 Linux/Windows GPU 或直接移植 Vulkan。

### 2. [Qualcomm Snapdragon GSR 2：移动时域超分基线](https://github.com/SnapdragonGameStudios/snapdragon-gsr)

直达：[SGSR2 shader](https://github.com/SnapdragonGameStudios/snapdragon-gsr/tree/main/sgsr/v2) · [Vulkan 样例](https://github.com/SnapdragonGameStudios/adreno-gpu-vulkan-code-sample-framework/tree/main/samples/sgsr2) · [Unreal 插件](https://github.com/SnapdragonGameStudios/snapdragon-game-plugins-for-unreal-engine) · [官方技术说明](https://www.qualcomm.com/developer/blog/2024/10/introducing-snapdragon-game-super-resolution-2)

**核心算法**

- SGSR2 是非神经网络 TAAU：先把低分辨率 color、depth、motion vector 转换为时域重投影需要的缓冲，再用历史帧重投影、历史有效性判断和当前帧信息重建高分辨率结果。
- 放大阶段使用为移动 GPU 优化的 Lanczos 类重建，并可追加锐化；官方同时给出两遍 compute、两遍 fragment 和三遍 compute 三种组织方式。
- fragment 版本把工作放进图形管线，尽量利用 tile/GMEM 驻留和固定功能采样；compute 版本则更便于异步调度和通用移植，两者算法目标相同但访存路径不同。

**复现平台要求**

- 官方 Vulkan framework 可构建 Android 和 Windows 版本：需要 Git、Python 3（官方测试 3.10.9）、CMake（官方测试 3.30+）和 Vulkan SDK 1.3+；Windows 使用 Visual Studio 2022。
- Android 还需要 Android Studio、Android SDK/NDK 和 Ninja；目标机以支持 Vulkan 的 Snapdragon/Adreno 为首选，其他 Vulkan GPU 可验证可移植性，但不能预期得到相同的 tile 优化收益。
- 引擎必须输出低分辨率 color、depth、motion vector、相机 jitter 和历史帧状态；应在透明/UI 合成之前执行，并正确处理相机切换和历史重置。

SGSR2 是专门为 Adreno tile-based GPU 优化的 TAAU。核心输入是低分辨率 color、depth 和 motion vector，提供 compute 两遍/三遍以及 fragment 两遍等变体，源码采用 BSD-3-Clause 许可。

Qualcomm 在 Snapdragon 8 Gen 3、1260×2800 输出下公布的数据很有架构价值：

| 放大倍率 | compute 两遍 | fragment 两遍 | compute 三遍 |
| ---: | ---: | ---: | ---: |
| 2.0× | 1.801 ms | 0.905 ms | 2.015 ms |
| 1.7× | 1.910 ms | 1.024 ms | 2.199 ms |
| 1.5× | 1.998 ms | 1.107 ms | 2.397 ms |

这里最重要的结论是：移动 GPU 上 compute 不一定优于 fragment。fragment 路径可以更好利用 tile/GMEM、固定功能插值和既有 render-pass 数据驻留。自研 GPU 应同时测 compute 与 fragment，不能只移植桌面 compute 版本。

建议实验：

1. 直接构建官方 Android Vulkan `sgsr2` 样例。
2. 固定场景，比较 native、bilinear、SGSR1、SGSR2 compute 和 SGSR2 fragment。
3. 扫描 1.5×、1.7×、2.0×，记录 ALU、texture、external read/write、tile load/store 与 GPU time。
4. 在透明、粒子、细线、镜面高光和快速相机移动场景检查 ghosting 与 flicker。
5. 将 SGSR2 的 pass 和中间纹理格式原样迁移到自研 GPU，作为第一套对照。

### 3. [Qualcomm AFME 与 Motion Estimation：硬件外推参考](https://github.com/SnapdragonGameStudios/adreno-gpu-opengl-es-code-sample-framework/tree/main/samples/amfe_power_saving)

直达：[AFME power-saving 样例](https://github.com/SnapdragonGameStudios/adreno-gpu-opengl-es-code-sample-framework/tree/main/samples/amfe_power_saving) · [Motion Estimation 样例](https://github.com/SnapdragonGameStudios/adreno-gpu-opengl-es-code-sample-framework/tree/main/samples/motion_estimation) · [Synchronous Space Warp 说明](https://www.qualcomm.com/developer/blog/2022/09/virtual-boost-vr-rendering-performance-synchronous-space-warp)

**核心算法**

- AFME 是帧外推而非双帧插值：应用交替提交真实渲染帧和 `glExtrapolateTex2DQCOM` 生成帧，驱动根据之前两张 color texture 推断运动并预测下一时刻画面。
- Motion Estimation 样例展示由 reference/target image 生成运动矢量纹理；AFME 把运动估计、遮挡处理和重建封装在 Qualcomm 驱动/硬件扩展中，应用侧主要负责纹理、同步和 present 节奏。
- 因为不等待未来帧，外推延迟低于插值，但新暴露区域没有真实未来信息，快速旋转、边界和非线性动画更容易出错。

**复现平台要求**

- 需要 Android Snapdragon/Adreno 设备，且 OpenGL ES 驱动实际暴露对应 QCOM motion-extrapolation 扩展；必须在运行时查询扩展，只有普通 OpenGL ES/Vulkan 支持并不够。
- 构建样例需要 Android Studio、Android SDK、NDK、Gradle 和 CMake，可按 Adreno OpenGL ES sample framework 的根目录构建脚本生成 APK。
- 公开代码只能复现 API 调用、交替帧渲染和功耗效果，不能复现专有运动估计内核；非 Qualcomm GPU 上应以 Mob-FGSR 外推或 ExtraNet 替代做算法对照。

AFME 样例每隔一帧停止传统渲染，通过 `glExtrapolateTex2DQCOM` 从之前的帧外推新帧，从而减少交替帧的 CPU/GPU 工作量。它比 SGSR2 更接近“芯片里增加插帧能力”的问题。

可观察的设计点：

- 由驱动扩展暴露 motion extrapolation，而不是让游戏实现全部算法。
- 插帧工作和 `eglSwapBuffers()`、present engine、CPU 提交节奏强相关。
- 可把省下的功耗预算用于降温，也可重新分配给游戏逻辑或图形特效。
- VR 路径使用外推而非等待未来帧的插值，以控制 motion-to-photon latency。

限制：AFME 核心是 Qualcomm 专有扩展，公开的是调用样例而不是硬件实现；是否可运行必须以目标设备驱动实际暴露的扩展为准。

### 4. [Arm ASR：由 FSR2 演化的移动时域超分](https://github.com/arm/accuracy-super-resolution)

直达：[Generic Library](https://github.com/arm/accuracy-super-resolution-generic-library) · [教程](https://documentation-service.arm.com/static/6814d88279c4c17ad8038ae0) · [官方介绍](https://developer.arm.com/mobile-graphics-and-gaming/arm-accuracy-super-resolution)

**核心算法**

- ASR 从 FSR 2.2.2 演化而来，是非神经时域超分：用当前低分辨率 color、depth、motion vector、jitter/exposure 和历史高分辨率结果完成重投影。
- 主要阶段包括深度/运动矢量重建与膨胀、disocclusion 和历史置信度判断、当前帧重建、时域累积以及可选锐化。
- Quality、Balanced、Performance 变体通过精度、采样和历史处理取舍降低带宽与算力，重点适配移动 tile-based GPU，而不是简单改变输入分辨率。

**复现平台要求**

- Generic Library 当前内置后端是 Vulkan；需要 C/C++ 工具链、CMake、Vulkan SDK，以及支持 compute shader 的 Vulkan GPU。可在 Linux/Windows 先构建，再部署到 Android Vulkan 设备。
- 最有代表性的目标是 Mali/Immortalis，但库并不把运行限制在 Arm GPU；跨 GPU 复现时应分别记录 shader time、external bandwidth 和 tile load/store。
- 引擎需提供 color、depth、motion vector、jitter、exposure/reactive 信息并管理历史资源；ASR 应在大多数显示分辨率后处理和 UI 之前接入。

Arm ASR 基于 AMD FSR 2.2.2，针对移动带宽和算力做了 Quality、Balanced、Performance 三档优化，并提供 generic library 和 Unreal 插件。

官方在 2400×1080 输出上给出的 Immortalis-G720 数据约为 2.0–2.9 ms；在 2800×1260 上约为 2.5–4.1 ms。它很适合作为“桌面 FSR2 移动化后需要做什么”的差分研究对象。

建议实验：

- 对 FSR2 与 ASR 做 shader/pass 级 diff，标出删减的纹理访问、精度和历史处理。
- 在同一 Vulkan 场景中比较 SGSR2、ASR 和 FSR2。
- 扫描质量 preset 与放大倍率，不要把算法质量档和缩放倍率绑定为一个变量。
- 在 Mali/Immortalis 设备上用 Arm Performance Studio 检查 external bandwidth 与 tile 行为。

### 5. [Arm NSS/NFRU：开放的移动神经超分与插帧](https://github.com/arm/neural-graphics-sdk-for-game-engines)

直达：[SDK 用户指南](https://github.com/arm/neural-graphics-sdk-for-game-engines/blob/main/docs/user_guide.md) · [训练与 QAT Notebook](https://github.com/arm/neural-graphics-model-gym-examples) · [数据集](https://huggingface.co/datasets/Arm/neural-graphics-dataset) · [NSS 模型](https://huggingface.co/Arm/neural-super-sampling) · [Vulkan ML 模拟层](https://github.com/arm/ai-ml-emulation-layer-for-vulkan)

**核心算法**

- NSS 是量化神经时域超分：融合低分辨率 color、depth、motion vector 和历史信息，通过 Vulkan tensor/data-graph 执行模型，输出高分辨率帧；Model Gym 提供训练、微调、QAT 和 VGF 导出链路。
- NFRU 是双帧插值：融合前后真实帧、引擎 motion vector、depth、内部 optical-flow correspondence 和神经网络，在两个真实帧之间生成一帧；固定增加一个真实渲染帧的等待延迟。
- NFRU 把 prepare、optical flow/data graph、warp/disocclusion 和 compose 暴露为可观察阶段，适合比较“全 GPU”“GPU + NPU”以及固定功能光流单元三种映射。

**复现平台要求**

- SDK 支持 Windows 11 x64、Linux x64 和 Android AArch64，图形后端为 Vulkan；需要 Python 3、CMake（SDK 文档限定 3.21–3.31）和 Vulkan SDK，官方推荐 1.4.321.0。
- Windows 需要 Visual Studio 2019+；Linux 需要 C++20 工具链；Android 需要 Android Studio 2024.2.1+、NDK r23+、Android SDK 和 JDK 21，官方验证环境为 Android 17。
- 原生执行依赖 `VK_ARM_tensors`、`VK_ARM_data_graph`，NFRU 还使用 optical-flow data-graph 扩展；设备尚未支持时需加载 Arm Vulkan ML emulation layer。当前图只支持量化 data graph，应重点验证 INT8 精度和模拟层开销。

这是当前最值得研究的开放神经路线。SDK 包含：

- NSS：神经时域超分；
- NFRU：结合引擎 motion vector、optical flow correspondence 和神经网络的插帧；
- Windows x64、Linux x64、Android AArch64 样例；
- Sponza 实时场景和 EXR 数据集模式；
- 训练、微调、评估、QAT 与 VGF 导出 Notebook；
- 在真实 Vulkan ML 扩展普及前使用的 graph/tensor emulation layers。

NFRU 生成两个真实渲染帧之间的一帧，采用固定的一个渲染帧额外延迟。其样例公开 color、depth、motion vector、optical-flow、disocclusion mask、warp 与输出等调试视图，非常适合建立芯片 workload 和中间张量清单。

建议实验：

1. 先在 Linux/Windows 用 Vulkan ML 模拟层运行 `nss` 和 `nfru` 样例。
2. 用 Model Gym 跑预训练模型评估，然后只做小规模 fine-tune。
3. 比较 FP16 与 QAT INT8 的画质、模型执行时间、激活带宽和峰值存储。
4. 将模型拆成 tensor、optical flow、warp/compose 三部分，分别映射 GPU、NPU 或固定功能单元。
5. 再尝试 Android；官方文档虽已提供 Android AArch64 路径，但目标设备是否具有原生 Vulkan ML 扩展必须单独查询，缺失时仍需模拟层。

### 6. [AMD FSR 3.1：最完整的开放桌面级参考](https://gpuopen.com/fidelityfx-super-resolution-3/)

直达：[FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) · [FSR 样例说明](https://gpuopen.com/manuals/fsr_sdk/samples/samples/super-resolution/) · [FSR2 独立源码](https://github.com/GPUOpen-Effects/FidelityFX-FSR2)

**核心算法**

- FSR 3.1 的超分部分是非神经 TAAU，使用低分辨率 color、depth、motion vector、jitter、exposure 和 reactive mask 做历史重投影、锁定/拒绝、累积与重建。
- Frame Generation 同时利用引擎 motion vector/depth 与 SDK optical flow，把前后两张显示分辨率真实帧重投影到中间时刻，再处理 disocclusion、融合和 UI composition；它是插值，必须等待后一张真实帧。
- proxy swapchain 和 frame-pacing 层负责在真实 present 之间插入生成帧；3.1 的 SR/FG 可解耦，便于只保留 FG 并换用其他 upscaler。

**复现平台要求**

- 最直接的官方复现环境是 Windows 10/11、DirectX 12 GPU、Visual Studio/CMake 和 FidelityFX SDK；FSR 3.1 也有 Vulkan 集成路径，适合在 Linux/Android 原型中移植 shader 与资源调度。
- SR 需要低分辨率 color/depth/MV、jitter、exposure/reactive mask；FG 还需要显示分辨率帧、depth/MV、HUD-less color 或独立 UI，以及 swapchain/present 控制权。
- 桌面样例可在 AMD、NVIDIA、Intel GPU 上做功能基线；移动 GPU 移植必须重新预算显示分辨率 optical flow、额外历史缓冲、异步队列和 UI 合成的带宽，不能直接套用桌面耗时。

FSR 3.1 把 temporal upscaling、optical flow、frame interpolation、swapchain 和 UI composition 连成完整流程，提供 HLSL/C++ 源码以及 DirectX 12、Vulkan 路径。它适合研究完整系统，而不是直接拿来当移动性能目标。

必须重点阅读的部分：

- upscaling 与 frame generation 已解耦，可替换不同 upscaler；
- frame generation 在 display resolution 上运行，带宽代价不能忽略；
- UI 必须独立合成、重新渲染或提供 HUD-less buffer；
- optical flow 和 interpolation 可放在 present queue 或 async queue；
- 官方建议先关闭 async，把画质和正确性验证好再并行化；
- 低于 30 FPS 的输入应避免，桌面官方建议以至少约 60 FPS 的生成前帧率获得较好体验。

截至调研日，AMD 已有更新的 FSR SDK/ML 路线，但 FSR 3.1 仍是更适合跨厂商学习和移动移植的开放基线。不要把仅适用于新 Radeon 硬件的最新 ML 路径误当成通用移动实现。

### 7. [NVIDIA Streamline/DLSS-FG：研究接口，不研究闭源内核](https://github.com/NVIDIA-RTX/Streamline)

直达：[集成样例](https://github.com/NVIDIA-RTX/Streamline_Sample) · [DLSS-FG 编程指南](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md) · [Optical Flow SDK](https://developer.nvidia.com/optical-flow-sdk)

**核心算法**

- DLSS Frame Generation 以当前/前一真实帧、depth、密集 motion vector、相机常量和 HUD-less color 为输入，结合 RTX optical-flow accelerator 的像素运动与引擎几何运动，再由专有 AI 网络生成中间帧。
- Streamline 负责资源 tag、frame token、插件调度和 swapchain/present 集成；Reflex 通过调整 CPU 提交和渲染队列降低插帧增加的端到端延迟。
- 网络结构、训练数据和 DLSS-G 推理内核闭源；可复现的是输入协议、UI/HDR/动态分辨率处理、时序同步与性能统计，不是算法权重。

**复现平台要求**

- 需要 Windows 10 20H1（build 19041）或更新版本、开启 Hardware-Accelerated GPU Scheduling、支持 DLSS Frame Generation 的 NVIDIA RTX GPU、匹配驱动和正式 `sl.dlss_g.dll`。
- 官方 Streamline sample 构建需要 CMake 3.20+、Visual Studio、Windows SDK 10.0.22000+；Vulkan 路径还需要 Vulkan SDK（样例基线 1.2.198.1+）和近期 DXC。
- 可选择 Direct3D 12 或 Vulkan 接入，并提供 depth/MV/HUD-less buffer 与正确 frame ID；Streamline 框架源码开放，但插件二进制及支持硬件不可替换，因此不适合在自研移动 GPU 上直接复现。

Streamline 框架、接口、样例和编程指南是开放的，但 `sl.dlss_g.dll` 是预编译插件，不能从仓库构建。因此它适合回答以下系统问题：

- 应用怎样提交 depth、motion vector、HUD-less color 和 camera constants；
- 何时重建或代理 swapchain；
- Reflex/低延迟模块怎样与 frame ID 和 present 对齐；
- UI、动态分辨率、HDR、VSync、VRR 和多 viewport 怎样处理；
- frame generation 的显存占用与显示节奏怎样统计。

它不适合作为自研算法或 RTL 的直接起点。购买 Jetson 也不能等价获得桌面 DLSS-FG 的开放实现；若目标是算法复现，FSR 3.1、Mob-FGSR 和 Arm NFRU 更合适。

### 8. [Intel XeSS 3：跨厂商接口与调试参考](https://github.com/intel/xess)

直达：[XeSS-FG 指南](https://github.com/intel/xess/blob/main/doc/xess_fg_developer_guide_english.md) · [基础 FG 样例](https://github.com/intel/xess/tree/main/samples/basic_sample_frame_generation)

**核心算法**

- XeSS-FG 在 present 前执行一组 compute pass，融合前后真实帧、当前到前一帧的 motion vector、depth 和 frame constants，通过专有 AI 插值内核合成中间帧。
- SDK 用 proxy `IDXGISwapChain4` 管理生成帧、UI 和 pacing；可提交 HUD-less color 或选择 UI texture/回调等合成模式。
- Xe Low Latency（XeLL）是强制组成，用于限制渲染队列和校准帧标识；算法二进制可跨厂商运行，但训练和网络实现不开放。

**复现平台要求**

- 需要 Windows 10 x64 build 19043+ 或 Windows 11 build 22000+、DirectX 12、XeSS SDK 动态库和 Microsoft Visual C++ Redistributable 14.40.33810+。
- Intel 路径要求文档列出的 Arc/Core Ultra GPU 及 32.0.101.7029+ 驱动；非 Intel GPU 需要 Shader Model 6.4，且只支持生成一张中间帧。
- 必须集成 XeLL 1.3+，并提供 motion vector、depth、frame constants、UI/HUD-less 资源及 swapchain 代理权限；因此适合桌面接口对照，不是 Android/移动 Vulkan 的直接复现平台。

XeSS 3 SDK 提供 SR、Frame Generation 和 Xe Low Latency。FG 以一系列 compute shader pass 加 proxy swapchain 实现，要求 motion vector、depth、frame constants，并强制配合 XeLL。非 Intel、支持 Shader Model 6.4 的 GPU也可运行一帧插值模式。

它的内核不是面向移动 GPU 的开放参考，但输入标注、UI 模式、debug inspector、present pacing 和跨厂商 fallback 设计值得纳入 API 对照表。

### 9. [Apple MetalFX：当前 M4 Pro 上的黑盒对照实验](https://developer.apple.com/documentation/metalfx)

直达：[Frame Interpolator API](https://developer.apple.com/documentation/metalfx/mtlfxframeinterpolatordescriptor) · [WWDC25 集成讲解与代码](https://developer.apple.com/videos/play/wwdc2025/211/) · [Apple Game Porting Toolkit 指南](https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/using-metalfx-frame-interpolation/SKILL.md)

**核心算法**

- MetalFX Temporal Scaler 使用低分辨率 color、depth、motion vector、jitter 和历史帧执行 Apple 专有时域/机器学习超分；Frame Interpolator 再读取前后真实 color、motion、depth，生成一张中间显示帧。
- 推荐在 tone mapping 后执行 frame interpolation；UI 可离屏提供给 interpolator、在生成帧后独立合成，或按显示帧重绘，present thread 负责均匀 pacing。
- 模型、训练方法和 kernel 闭源，只能把 MetalFX 当作 API、画质、时延、内存和功耗的黑盒对照。

**复现平台要求**

- 需要支持 MetalFX Frame Interpolation 的 Apple silicon、相应 macOS/iOS/iPadOS/tvOS/visionOS SDK、Xcode 和 Metal 工程；创建前用 `supportsDevice` 查询，不能只按产品名称假定支持。
- 当前 M4 Pro 可作为首选本地平台；应用需创建符合 descriptor 格式/尺寸约束的 color、depth、motion、output 和可选 UI texture，并把 scaler/interpolator 与 present thread 串起来。
- 无法在 Android、Vulkan 或非 Apple GPU 上运行，也无法提取算法权重；跨平台实验应固定相同场景、输入/输出分辨率和画质指标，只比较可观察行为。

当前 M4 Pro 无需购买开发板即可测试 MetalFX temporal upscaling 与 frame interpolation。Frame Interpolator 使用前后 color、motion、depth 和输出纹理，可与 temporal scaler 串联；Apple 还公开了 UI 与 present-thread 的集成指导。

其价值是建立 Apple 移动/统一内存架构上的黑盒性能和接口基线。算法本身闭源，Metal API 也不能直接迁移到 Android/Vulkan，因此应把它作为对照组，而不是主实现。

### 10. [ExtraNet 与通用视频插帧：算法补充，不做第一主线](https://github.com/fuxihao66/ExtraNet)

直达：[ExtraNet 源码](https://github.com/fuxihao66/ExtraNet) · [Google FILM](https://github.com/google-research/frame-interpolation)

**核心算法**

- ExtraNet 做未来帧外推：利用延迟渲染/交错帧产生的 G-buffer 与遮挡运动信息，把 backward motion vector splat 成近似 forward/occlusion motion，再由神经网络预测未来 shading/base color 并修复 disocclusion。
- FILM 是纯图像双帧插值：共享卷积权重的多尺度特征提取器在不同分辨率建立对应关系，再合成中间帧；训练只需图像 triplet，不依赖预训练 optical-flow/depth 网络。
- ExtraNet 更贴近低延迟游戏外推和引擎协作；FILM 更适合建立“无 depth/MV 的视频算法”画质下限/上限对照，不应与引擎感知方案混作同一条件。

**复现平台要求**

- ExtraNet 训练/离线推理需要 Python 与仓库 requirements；数据生成依赖其定制 Unreal Engine 4.25.3 或更早版本、Windows/DX12，并按说明开启 ray tracing、关闭后处理及不受支持的透明/粒子等内容。
- ExtraNet 的实时性能参考另有 TensorRT 实现，需 NVIDIA CUDA/TensorRT GPU 环境；公开 UE 代码更适合生成训练数据和离线验证，不能直接视作成熟游戏插件。
- FILM 最容易通过官方 TensorFlow 2/Colab Notebook 复现；本地可用 Linux/Windows Python + TensorFlow，GPU 对高分辨率序列更实用，CPU 仅适合小图功能验证。它不需要开发板、depth 或 motion vector。

ExtraNet 研究低延迟 frame extrapolation，并提供数据生成、训练和 TensorRT 推理材料；适合研究神经外推和 disocclusion motion vector。FILM、RIFE 等视频插帧方案只依赖图像，适合学习 optical flow 和遮挡合成，但通常不知道游戏引擎的 depth、object motion、camera jitter 和 UI，因此不能代表游戏帧生成的最佳上限。

## 不购买开发板也能完成的实验

### 实验 A：离线数据与画质基准

资源：[Arm Neural Graphics 数据集](https://huggingface.co/datasets/Arm/neural-graphics-dataset) · [Mob-FGSR 数据与代码](https://github.com/Mob-FGSR/MobFGSR) · [NVIDIA FLIP](https://github.com/NVlabs/flip)

工作内容：

1. 统一保存 native high-resolution ground truth、low-resolution color、depth、motion vector、exposure、jitter 和 UI mask。
2. 为中间时刻离线渲染真实 ground-truth frame，不要把相邻帧简单平均当真值。
3. 统一运行 bilinear、SGSR1、SGSR2/ASR、Mob-FGSR、NSS/NFRU。
4. 输出 PSNR、SSIM、LPIPS、FLIP、temporal warping error 和 flicker 曲线。
5. 保存 error map，而不只保存单一平均分。

最低硬件：现有电脑即可完成数据检查和指标；Mob-FGSR 原始运行需要 OpenGL 4.3 的 Linux/Windows GPU。

### 实验 B：当前 M4 Pro 的 MetalFX 黑盒实验

工作内容：

- 构造一个含静态背景、动态物体、透明粒子、UI 和快速相机移动的小型 Metal 场景。
- 依次运行 native、spatial upscale、temporal upscale、frame interpolation、upscale + interpolation。
- 用 Metal GPU capture 和 Performance HUD 记录各阶段时间、内存和 present cadence。
- 故意反转 motion vector、改变 scale、错误设置 exposure，建立输入错误到伪影的映射。

这套实验能快速理解接口，但不能测 Android tile GPU，也不能看到 MetalFX 内核。

### 实验 C：桌面 FSR 3.1、DLSS-FG、XeSS-FG 对照

最低硬件：Windows 11 + D3D12 GPU；DLSS-FG 另需支持的 RTX GPU，XeSS-FG 的完整能力优先 Intel Arc。

工作内容：

- 用各自官方 sample 对齐 1080p/1440p、1.5×/1.7×/2×。
- 统一关闭 motion blur，先关闭 async compute。
- 检查 UI 分离、HUD-less color、camera cut、动态分辨率和 VRR。
- 将测试目的限定为系统接口、画质上限和 pacing，不把桌面耗时直接外推到移动 GPU。

### 实验 D：Arm NSS/NFRU 模拟层

在没有新一代神经 GPU 的情况下，可在 Linux/Windows 使用 Vulkan ML emulation layers 跑 Sponza 和 EXR dataset mode，并用 Model Gym 做训练、fine-tune、QAT 和导出。这是购买硬件之前研究 neural workload、tensor shape、激活生命周期和算子支持的最佳入口。

## 设备和开发板选择

| 设备 | 能做什么 | 优点 | 限制 | 建议 |
| --- | --- | --- | --- | --- |
| 现有 Apple M4 Pro | MetalFX temporal SR + frame interpolation | 无需购买，接口完整 | 内核闭源、Metal 专用 | 立即做黑盒和接口实验 |
| Snapdragon 8 Gen 3/8 Elite Android 手机 | SGSR2、Vulkan 样例、Mob-FGSR 移植、部分 AFME | 真实屏幕、Android compositor、温控和功耗最接近移动产品 | root/counter/驱动扩展权限因机型而异 | 第一台外部移动测试设备 |
| [Thundercomm C8550 Dev Kit](https://www.thundercomm.com/product/c8550-development-kit/) | Adreno 740、Android、Vulkan、外接显示与稳定供电 | 接口多，便于持续 profiling 和自动化 | 当前售价约 1,599 美元；SoC 是 QCS8550/8 Gen 2 同级，不代表最新手机 | 需要实验室自动化时再买 |
| [Qualcomm RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit) | QCS6490、Adreno 642L、Android/Linux/Vulkan | 官方开发板，适合低端压力与兼容测试 | GPU 较老；不能假设支持 AFME 专有扩展 | 做 portability/低带宽档，不做峰值代表 |
| Immortalis-G720/G925 Android 手机 | Arm ASR、Arm Performance Studio、NSS/NFRU 兼容性研究 | 最适合验证 Arm 移动优化与 tile GPU counters | 地区/机型 SoC 版本需核实；新 Vulkan ML 扩展未必存在 | 第二类跨架构对照设备 |
| Windows RTX/Radeon/Arc PC | FSR 3、DLSS、XeSS 完整 reference sample | 工具和样例成熟 | 功耗、内存体系与移动 GPU差异大 | 只做参考上限和接口研究 |
| Jetson Orin | TensorRT 光流/神经模型原型 | CUDA/TensorRT 方便 | 不是手机 GPU；不等于能运行桌面 DLSS-FG | 不为本项目单独购买 |
| FPGA | 自定义 warp/flow/tensor 单元 | 可验证数据通路和 RTL | 接入真实 Android 图形栈、DDR 和 display pacing 成本高 | 算法和 workload 稳定后再做 |

对于这一课题，真实 Android 手机往往比普通开发板更有价值，因为插帧是否成功不仅取决于 shader 时间，还取决于 display compositor、VSync/VRR、swapchain、触控延迟、DVFS、温升和持续功耗。开发板更适合自动化、外接功耗仪和长时间 trace。

## 推荐的八周实验路线

### 第 1 周：建立统一数据格式

- 定义 color、depth、motion vector、exposure、jitter、reactive mask、UI、frame ID 和 camera matrices。
- 从一个可控场景导出 native HR、low-res 和中间时刻 ground truth。
- 建立 scene cut、动态分辨率和历史 reset 标记。

验收：同一序列能被多个算法读取，motion vector 方向、单位和 scale 有自动检查。

### 第 2 周：空间与时域超分基线

- 跑 bilinear/bicubic、SGSR1、SGSR2、Arm ASR 或 FSR2。
- 扫描 1.5×、1.7×、2×。
- 比较静态、运动、遮挡、透明和粒子场景。

验收：输出画质指标、error map、单 pass 时间和外部内存流量。

### 第 3 周：Mob-FGSR 插值和外推

- 跑 I、E、SR、ISR、ESR 五种模式。
- 拆分 motion splat、hole filling、warp、blend、LUT SR。
- 比较 30→60、45→90、60→120 三个目标。

验收：明确插值和外推的画质—延迟曲线，不只报告显示 FPS。

### 第 4 周：Android Vulkan 移植

- 先跑 Qualcomm SGSR2 Vulkan sample。
- 把 Mob-FGSR 的关键 pass 迁到同一个 Vulkan harness。
- 同时保留 compute 与 fragment 两条路径。

验收：可在目标手机或内部开发板一键切换 native、SGSR2、Mob-FGSR，并导出 trace。

### 第 5 周：神经路线

- 跑 Arm NSS/NFRU 预训练模型和样例。
- 使用 Model Gym 完成一次小规模 fine-tune 和 QAT。
- 对比 FP16/INT8、GPU/NPU/模拟层。

验收：列出每层算子、tensor shape、权重、激活峰值、执行位置和不支持算子。

### 第 6 周：GPU/芯片 profiling

- 用 [Android GPU Inspector](https://developer.android.com/agi/frame-trace/frame-profiler)、[Arm Performance Studio](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio)、Snapdragon Profiler 或内部工具抓取 counters。
- 统计 ALU、texture、atomic、occupancy、cache、external bandwidth、tile load/store、async overlap。
- 测试 FP32、FP16、R11G11B10、RGBA8、RG16F 等格式。

验收：得到各 pass 的 compute-bound/memory-bound 分类和带宽账本。

### 第 7 周：显示、低延迟和 UI

- 接入独立 UI 合成、loading/menu bypass、camera cut reset。
- 测 VSync、VRR、frame limiter、swapchain image 数和 present pacing。
- 测输入到显示延迟；插值与外推分开统计。

验收：无周期性重复帧、长短帧交替、UI 扭曲和历史未重置问题。

### 第 8 周：持续功耗与架构提案

- 进行至少 20–30 分钟持续测试，记录温度、频率、功耗、FPS 和 1% low。
- 比较 native、低分辨率 + SR、低分辨率 + SR + FG。
- 根据 profiling 决定优先加速 optical flow、warp/splat、tensor、tile memory 还是 presentation/pacing。

验收：形成一份带 Pareto 曲线的架构建议，而不是只给出“帧率提高百分比”。

## 统一测试场景

至少包含以下场景，否则平均 PSNR 很容易掩盖真实问题：

- 栅栏、电线、草叶、头发等亚像素细线；
- 高速横移和快速旋转相机；
- 前景物体突然露出背景的 disocclusion；
- 动态物体与相机同时运动；
- 透明粒子、烟雾、火焰、玻璃和 alpha-tested foliage；
- 镜面高光、闪烁光源、阴影变化和曝光突变；
- UI、文字、准星、血条和半透明菜单；
- motion blur 开关；
- camera cut、传送和动态分辨率改变；
- 低输入帧率、帧时间抖动和热降频。

## 必须记录的指标

### 画质

- PSNR、SSIM：便于和论文对齐，但不能单独作为结论；
- LPIPS：感知差异；
- [FLIP](https://github.com/NVlabs/flip)：面向 rendered image 的可视化 error map；
- temporal warping error、flicker、ghost trail length；
- disocclusion、透明物、UI 等区域的分类指标。

### 性能与功耗

- 每个 pass 的 GPU timestamp；
- ALU/TEX/atomic 利用率与 occupancy；
- 片外读写字节、cache hit、tile load/store；
- 峰值临时内存和 history buffer；
- 平均功耗、energy/displayed frame、温升和降频时间；
- sustained FPS、1% low 和 present interval 方差。

### 延迟

- input sampling → simulation → render → SR → FG → present → display；
- rendered FPS 与 displayed FPS 分列；
- 插值引入的等待帧延迟；
- 外推的预测误差和 latency benefit；
- UI 是按真实帧率还是显示帧率更新。

## 对移动 GPU 架构的直接启示

以下是依据公开方案作出的架构推论，需要用上述实验验证：

1. **优先解决带宽，而不只是增加 MAC。** 1080p RGBA8 单张就是约 8.3 MB；多个全分辨率 history、flow、warp 和 output 每帧反复读写，很快达到数 GB/s。
2. **保留 fragment/tile 路径。** SGSR2 的公开结果说明，在 tile-based GPU 上 fragment 两遍可能显著胜过 compute。
3. **增加低成本 warp、splat、gather 与原子支持。** Mob-FGSR 的关键热点比通用大模型 GEMM 更像图像重采样和冲突消解。
4. **让 GPU、神经单元和显示单元直接共享资源。** 若中间 tensor/flow 必须回写系统内存，专用神经单元节省的算力可能被搬运抵消。
5. **把 motion vector、depth、UI 和 frame ID 变成稳定 ABI。** DLSS、FSR、XeSS、MetalFX 与 NFRU 都依赖一致的资源标注和历史管理。
6. **显示队列和低延迟是功能的一部分。** proxy swapchain、present pacing、VRR、frame limiter 和触控采样不能等算法完成后再补。
7. **插值与外推最好同时保留。** 普通游戏可偏向插值画质，VR/高交互场景可偏向外推延迟。
8. **支持可诊断中间输出。** motion、depth、disocclusion、reactive、optical flow、warped frames 和 UI mask 的 debug view 会显著缩短集成周期。

## 常见误区

- 不要把视频插帧模型的效果直接等同于游戏插帧；视频模型通常拿不到引擎 motion vector、depth 和 camera state。
- 不要只看显示 FPS；同时报告真实渲染 FPS、延迟和 frame pacing。
- 不要从静态截图判断时域算法；ghosting、flicker 和 judder 只能在序列中判断。
- 不要把 UI 放进普通插值路径后就宣布完成；UI 需要独立策略。
- 不要在一开始就启用 async compute；先验证输入和画质，再做并行调度。
- 不要用平均功耗代替持续温控测试；移动 GPU 的频率和温度会改变 10–30 分钟后的结论。
- 不要假设某 SoC 产品页写有 AFME 就意味着第三方应用一定能直接调用；以驱动扩展和 SDK 权限为准。
- 不要先买 FPGA；先从公开 shader、trace 和 tensor workload 找到真正值得固定功能化的热点。

## 最终优先清单

如果只做六套实验，建议按以下顺序：

1. [Mob-FGSR](https://github.com/Mob-FGSR/MobFGSR)：最贴近移动 GPU 超分插帧联合目标。
2. [Qualcomm SGSR2 Vulkan Sample](https://github.com/SnapdragonGameStudios/adreno-gpu-vulkan-code-sample-framework/tree/main/samples/sgsr2)：最容易开始的移动时域超分。
3. [Qualcomm AFME Sample](https://github.com/SnapdragonGameStudios/adreno-gpu-opengl-es-code-sample-framework/tree/main/samples/amfe_power_saving)：研究驱动/硬件外推接口。
4. [Arm ASR](https://github.com/arm/accuracy-super-resolution-generic-library)：研究 FSR2 如何移动化。
5. [Arm NSS/NFRU](https://github.com/arm/neural-graphics-sdk-for-game-engines)：研究开放神经超分、光流和插帧协同。
6. [AMD FSR 3.1](https://gpuopen.com/fidelityfx-super-resolution-3/)：研究完整桌面级系统、UI、swapchain 和 pacing。

购买顺序建议是：先用现有 M4 Pro 和一台 Windows GPU 主机完成离线与桌面对照；然后采购或借用一台 Snapdragon 8 Gen 3/8 Elite 手机；只有需要持续自动化、外接功耗仪和稳定 IO 时，再购买 C8550 或 RB3 Gen 2 开发板。

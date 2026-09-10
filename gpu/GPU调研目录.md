# GPU 调研

- [GPU Tensor FP8 专题](fp8-tensor/GPU-Tensor中的FP8：A20-Pro、超分插帧与神经图形.md)：A20 Pro 官方发布核实、超分插帧的精度与预算、Apple MLX / NVIDIA RTXNTC 源码，以及 FP8 与 INT8 的设计取舍；附数值与存储复算脚本。
- [Apple AFM / ASTC 论文深读](compressed-weights/apple-afm-astc-explained.md)：从权重数值、块编码、误差传播到低秩补偿，解释为什么能压缩、何时影响答案，以及公开实现的边界。
- [Compressed Weights 专题](compressed-weights/GPU压缩权重：Arm与高通、NVIDIA、Apple对比.md)：Mali G2-Ultra NX 官方证据、Arm 历史硬件与 Vela 编码器、模型到运行时的数据链路，以及高通 LPBQ、NVIDIA 稀疏/LUT、Apple ANE/ASTC 对照。
- [Motion Engine 专题](motion-engine/Motion-Engine专题：Arm光流加速、计算量与硬件取舍.md)：Arm 发布与 NFRU 源码、SAD/MAC 计算量、可能的硬件实现，以及 NVIDIA/高通/苹果对照；附复算脚本。
- [Arm 移动 GPU 的 AI / 神经网络能力演进](arm-ai/Arm移动GPU的AI与神经网络能力演进.md)：从 Midgard/Bifrost 通用计算、INT8 dot、Valhall、矩阵指令与 Vulkan ML，到 2026 Mali G2-Ultra NX 专用 Neural Accelerator；含演讲、论文、GitHub、Hugging Face 与芯片评估建议。
- [超分与插帧](超分插值/超分与插帧调研索引.md)：GPU 厂商神经超分/帧生成对照，以及移动 GPU 可复现实验路径。

# NPU / AI 加速器课程调研

调研日期：2026-08-31（Asia/Shanghai）

本目录整理 Stanford、MIT 与 Harvard 三门和 AI 加速器、边缘推理及软硬件协同相关的课程。材料只收录无需登录即可访问的公开页面和 PDF；每个子目录都记录了来源、材料年份、文件校验值及“最新”的判断依据。

## 先说结论

用户给出的三个条目中，有两个名称或编号需要更正：

| 学校 | 核实后的课程 | 最新公开学期 | 定位 | 公开 PDF 情况 |
| --- | --- | --- | --- | --- |
| Stanford | EE292D / CS329E — Machine Learning on Embedded Systems | 课程站与实验仓库更新至 2025 | 面向 Raspberry Pi、Jetson、Edge TPU 等资源受限设备的端侧 ML；项目制 | 2025 课程站不提供 PDF；保存了目前仍可公开下载、明确包含 EE292D 说明的 2023 官方 EE 课程目录 PDF |
| MIT | 6.5930/6.5931（原 6.888）— Hardware Architecture for Deep Learning | Spring 2026 | 从 DNN 算子、数据流映射到稀疏架构、先进器件和数值精度的加速器体系结构 | 2026 课程站公开完整讲义与 recitation PDF，已全部归档 |
| Harvard | CS 2420（原 CS 242）— Computing at Scale | Fall 2025 | 跨核心/节点的 CNN、Transformer 高效执行以及模型—算法—系统协同设计 | 最新课程只公开文字说明，材料库需 Harvard 账号；保存了教师官网仍公开提供的 2016 课程概览 PDF，并明确标为历史材料 |

## 三门课如何互补

- Stanford 更靠近端侧产品落地：模型选择、微调、量化、蒸馏、MLOps 与 Raspberry Pi 实验。
- MIT 更靠近 NPU 微架构设计：张量计算、数据流与映射、循环/分块、融合、稀疏性、数据搬移和精度。
- Harvard 更靠近规模化系统协同：systolic array、低比特量化、分布式训练/推理、MoE、推测解码、调度、安全与边云协同。

建议学习顺序是 Stanford 建立端侧部署直觉，MIT 深挖硬件内核与性能模型，再用 Harvard 的系统视角覆盖多节点、模型优化和数据中心/边缘的共同问题。

## 目录

- [`stanford-ee292d/`](stanford-ee292d/)：课程身份核验、课程结构、端侧 NPU 相关性与公开材料。
- [`mit-6.5930/`](mit-6.5930/)：Spring 2026 课程说明及全套公开 PDF。
- [`harvard-cs2420/`](harvard-cs2420/)：编号演变、Fall 2025 课程说明及公开 PDF 的可用性边界。
- [`lab-survey/`](lab-survey/)：神经网络、Transformer 与 AI 加速器实践 Lab 调研；先列无需开发板的路线，再列需要 Raspberry Pi、Arduino、Jetson 或 FPGA 的路线。

## 口径与限制

“最新 PDF”指截至调研日，从课程官方站、学校目录或教师官方页面无需登录即可下载的最新相关 PDF，不把搜索结果的新旧排序当成课程版本。若最新课程只提供 HTML 或把材料放在 Canvas/受限数据库中，本目录不会绕过权限，而会保存能合法公开取得的最近 PDF，并在对应 README 中注明年份差距。

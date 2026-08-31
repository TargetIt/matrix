# Harvard CS 2420（原 CS 242）

## 编号与最新课程

- 官方名称：**Computing at Scale**。
- 教师：H. T. Kung。
- 课程在 Fall 2023 仍使用 CS 242，Fall 2024 起教师官方页使用 CS 2420。
- 截至 2026-08-31，最近一次公开课程说明是 Fall 2025。

Fall 2025 聚焦降低 AI 训练与推理成本：在多核心或多节点上实现 CNN 与 Transformer，并协同设计模型、数据整理、计算算法和系统架构。列出的技术包括 systolic array、低比特运算、剪枝、量化、蒸馏、LoRA、MoE 动态路由、推测解码、Stable Diffusion 合成数据、数据/模型安全、内存访问调度、强化学习推理和 test-time compute。

用户描述中的“Machine Learning Systems”更像课程内容标签，而不是这门课的官方副标题；“边缘 NPU”是课程覆盖的尺度之一，但 Fall 2025 说明同时覆盖多核、多节点和大规模推理，不能把它概括为纯移动端部署课。

## 与 NPU 的关系

Harvard 的优势是把 NPU 放进完整系统中评估：阵列结构与低精度只是起点，模型压缩、内存调度、分布式并行、服务吞吐、安全和数据生成都会改变加速器的真实收益。它适合回答“某种 NPU 优化放到端到端系统后是否仍然有效”。

## PDF 可用性说明

最新 Fall 2025 页面只公开课程文字说明。教师页面明确说明课程材料数据库需要 Harvard 邮箱/账号，公开页面没有列出 2024 或 2025 讲义 PDF。

`pdf/cs242-course-overview-2016.pdf` 是教师官方课程页仍公开链接的最近一份 CS 242 PDF，内容是 2016-01-25 的课程概览。它用于追溯课程关于大规模计算、CPU/GPU/FPGA/ASIC 映射、通信与数据复用的基础框架，不应当被当作 Fall 2025 syllabus。最新课程内容以本 README 引用的官方 HTML 为准。

## 官方来源

- H. T. Kung 课程页（含 Fall 2025、Fall 2024、Fall 2023 及更早说明）：https://www.eecs.harvard.edu/htk/courses/
- 历史 PDF 原始地址：https://s3.amazonaws.com/harvardcs242/2016-01-25courseoverviewMondaynoon.pdf

下载日期、文件大小、页数与 SHA-256 见 [`SOURCES.md`](SOURCES.md)。

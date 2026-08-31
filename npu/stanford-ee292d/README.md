# Stanford EE292D / CS329E

## 核实结果

- 官方名称：**Machine Learning on Embedded Systems**。
- 交叉编号：EE292D / CS329E。
- 教学团队公开页列出 Zain Asgar、Pete Warden、Sachin Katti、Ankita Nayak。
- 课程站及配套实验仓库在 2025 年仍有更新；它不是名为 “Projects in AI and Machine Learning Hardware” 的课程。

课程是项目制端侧机器学习课，目标平台包括 Arduino、Raspberry Pi、Jetson 和 Edge TPU。内容覆盖模型训练/微调、量化、PEFT/LoRA、知识蒸馏、DNN 硬件、MLOps，最终项目占总评 50%。公开实验目前以 Raspberry Pi 5 为主要平台，涉及本地 LLM、图像分类/检测、语音识别、翻译和微调部署。

## 与手机 NPU 的关系

课程不直接讲手机 SoC 中 NPU 的 RTL 或物理实现，但非常适合建立端侧 NPU 的“上层约束”意识：模型必须在有限功耗、内存、带宽和时延预算内运行；量化、蒸馏与 PEFT 会直接影响算子类型、精度格式、片上存储和编译器需求。它更偏部署与软硬件边界，而不是加速器微架构。

## PDF 材料

`pdf/stanford-ee-catalog-2023-ee292d.pdf` 是 Stanford 官方 2023 Electrical Engineering 课程目录，包含 “EE292D - Embedded Systems” 条目及课程说明。文件名保留年份，避免误认为它是 2025 讲义。

截至 2026-08-31，2025 版课程站和官方实验仓库没有提供 syllabus/lecture PDF；最新内容以 HTML 和 Jupyter Notebook 发布。因此，本目录没有把网页转印件伪装成官方 PDF，也没有收录已失效的第三方学生报告链接。

## 建议重点

1. Week 4：量化、数据格式、QAT。
2. Week 5：PEFT、LoRA、adapter、知识蒸馏。
3. Week 6：DNN 加速器设计如何影响模型设计与性能。
4. Labs 1–6：从模型运行到微调、优化和端侧部署的完整链路。

## 官方来源

- 课程站：https://ee292d.github.io/
- 公开实验仓库：https://github.com/ee292d/labs
- Stanford Bulletin 的交叉编号课程页：https://bulletin.stanford.edu/courses/2213942
- 归档 PDF 原始地址：https://coursedog-pdfs-public-prod.s3.us-east-2.amazonaws.com/stanford/catalog-pdf/bdtfOgn3HZQ0sWNriOjM/MH7UwGM8cXPmBy35Kr4vELUne/ELECTENGR-2023-04-05-20-46-46.pdf

下载日期、文件大小、页数与 SHA-256 见 [`SOURCES.md`](SOURCES.md)。

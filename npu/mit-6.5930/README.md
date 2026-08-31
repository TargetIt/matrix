# MIT 6.5930 / 6.5931（原 6.888）

## 课程概况

- 官方名称：**Hardware Architecture for Deep Learning**。
- 最新公开课程：Spring 2026。
- 教师：Vivienne Sze、Joel Emer。
- 先修：信号处理或机器学习，以及计算机组成/计算结构或同等背景。

这三门课中，MIT 6.5930/1 对 NPU 微架构最直接。课程从 DNN 组件和张量表示出发，逐步进入数据流映射、分块、融合、稀疏性、稀疏架构、先进器件、数值精度及数据搬移分析，随后以持续里程碑的团队项目收尾。

## Spring 2026 主线

1. DNN 算子与应用：理解工作负载，而不是只看峰值 TOPS。
2. Mapping / partitioning / fusion：决定 PE 阵列利用率、数据复用和片外访存量。
3. Sparsity：从稀疏表示到能够真正节省周期与能耗的硬件结构。
4. Advanced technologies：研究新存储/计算技术对架构的约束。
5. Numeric precision：低精度格式、精度—吞吐—能效权衡。
6. Data movement：以形式化工具和性能模型计算搬移开销。
7. Final project：将架构观点落实为可测量、可比较的设计。

## PDF 材料

`pdf/` 收录 Spring 2026 syllabus 页面截至调研日公开链接的全部 17 个 PDF：14 份 lecture PDF（L07 同时有 fusion 与 sparsity 两份）、1 份项目说明和 2 份 recitation。网页还包含 HTML slides、视频、Notebook 和课堂仓库链接，这些不属于 PDF，未复制进来。

## 官方来源

- 课程主页：https://csg.csail.mit.edu/6.5930/
- Spring 2026 syllabus：https://csg.csail.mit.edu/6.5930/syllabus.html
- Reading list：https://csg.csail.mit.edu/6.5930/readinglist.html
- 旧编号 6.888 的 MIT EECS 说明：https://eecsis.mit.edu/academic-information.html

逐文件原始 URL、页数、大小和 SHA-256 见 [`SOURCES.md`](SOURCES.md)。

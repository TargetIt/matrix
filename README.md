# matrix

交互式 CUDA SGEMM 教程与来源可视化。

## 三篇来源专题

每个专题目录都包含以下文件：

- `SOURCE.md`：NotebookLM 笔记本、带 `[高]` 的来源标题、Source ID 和 URL。
- `notebooklm-export.md`：NotebookLM CLI 的原始导出证据。
- `original.md`：清理网页 chrome 后的可读原文。
- `index.html`：可直接运行的交互可视化。

- `cutlass-hierarchy/`：CUTLASS GEMM 层级、tile policy 与双缓冲流水。
- `sgemm-practice/`：RTX 3090 上七个 SGEMM kernel 的优化阶梯。
- `cuda-matmul-worklog/`：A6000 性能演进、Roofline、计算下界与 occupancy。

`sgemm-practice` 是特殊情况：NotebookLM 对 GitHub URL 的提取主要返回站点
导航页面，因此其 `notebooklm-export.md` 保留原始抓取作为证据，
`original.md` 使用同一上游仓库的 README 正文。详细决策见该目录的
`SOURCE.md`。

## CUDA SGEMM 教程

- 打开 `./index.html` 查看统一入口、教程大纲、前言与步骤 1 至步骤 5。
- 根入口提供“步骤与原文索引”，每个独立演示顶部也可直接返回对应原文。
- 独立基础演示位于 `./demos/`，可直接打开对应 HTML 文件运行。
- 六个教程演示依次覆盖内存层级、Naive SGEMM、合并访存、SMEM Block
  Tiling、向量化访存以及 Warp Tiling/CUTLASS 层级。

根入口将 6 个教程步骤演示与 3 个来源专题分区展示。教程演示用于建立连续的
学习路径，来源专题用于保留文章数据和进行更深入的参数探索。

本仓库全部页面为静态 HTML，无需安装依赖。为避免浏览器对本地文件的限制，
也可以在仓库根目录执行：

```bash
python3 -m http.server 4173
```

## 质量与回归

- 需求、数据等级和视觉语义：`doc/requirements.md`
- 验证方法：`doc/verification.md`
- 来源文件完整性：`doc/source-manifest.json`
- 遗留问题：`doc/legacy_issues.md`

一键回归：

```bash
./scripts/test.sh
```

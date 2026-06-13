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

## 原有教程

- 打开 `./index.html` 查看统一入口、教程大纲、前言、步骤 1 与步骤 2。
- 独立基础演示位于 `./demos/`，可直接打开对应 HTML 文件运行。

根入口中的 3 个基础演示来自仓库既有内容；本次新增交付是上述 3 个来源专题，
两者在入口中分区展示。

本仓库全部页面为静态 HTML，无需安装依赖。为避免浏览器对本地文件的限制，
也可以在仓库根目录执行：

```bash
python3 -m http.server 4173
```

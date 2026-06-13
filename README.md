# matrix

交互式 CUDA SGEMM 教程与来源可视化。

## 三篇来源专题

每个专题目录都包含 `original.md` 原文和可直接运行的 `index.html`：

- `cutlass-hierarchy/`：CUTLASS GEMM 层级、tile policy 与双缓冲流水。
- `sgemm-practice/`：RTX 3090 上七个 SGEMM kernel 的优化阶梯。
- `cuda-matmul-worklog/`：A6000 性能演进、Roofline、计算下界与 occupancy。

`sgemm-practice/notebooklm-capture.md` 额外保留 NotebookLM 对 GitHub 页面
的原始抓取；该专题的 `original.md` 使用上游仓库 README 正文。

## 原有教程

- 打开 `./index.html` 查看统一入口、教程大纲、前言、步骤 1 与步骤 2。
- 独立基础演示位于 `./demos/`，可直接打开对应 HTML 文件运行。

本仓库全部页面为静态 HTML，无需安装依赖。为避免浏览器对本地文件的限制，
也可以在仓库根目录执行：

```bash
python3 -m http.server 4173
```

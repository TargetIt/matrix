# Digest 维护说明

本目录同时保留 Markdown 和 HTML 两种版本：

- Markdown 是后续维护的主版本，适合手机阅读和直接编辑。
- HTML 是网页展示版本，后续应从 Markdown 内容同步更新。
- 当前主 Markdown 入口是 [index.md](index.md)。

## 主文档

- [CUDA MMM Worklog 中文导读](cuda-mmm.md)
- [NVIDIA_SGEMM_PRACTICE 中文导读](sgemm-practice.md)
- [CUTLASS Linear Algebra 中文导读](cutlass.md)

## 同步规则

后续修改内容时，先更新对应 Markdown，再同步到同名 HTML：

- `cuda-mmm.md` → `cuda-mmm.html`
- `sgemm-practice.md` → `sgemm-practice.html`
- `cutlass.md` → `cutlass.html`

旧文件 `cuda-matmul-worklog.md` 和 `cutlass-hierarchy.md` 暂时保留，避免破坏历史链接；新内容以本 README 列出的主文档为准。

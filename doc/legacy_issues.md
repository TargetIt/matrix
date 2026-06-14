# 遗留问题

本文件记录尚未完成、暂缓处理或需要后续验证的事项。完成一项后必须勾选，
并补充完成提交和验证结果；新增需求如果未在当次交付中闭环，也必须登记在此。

## 当前未完成

- [ ] `LEGACY-001` 教程步骤 3：Shared Memory Caching & Block Tiling
  - 当前状态：根入口只有大纲，没有详细教程和独立交互演示。
  - 需要交付：步骤原理、GMEM 到 SMEM 的协作加载、同步、tile 复用和边界处理讲解。
  - 演示建议路径：`demos/step3-shared-memory-block-tiling.html`
  - 验收条件：可调 block/tile 参数；可视化 GMEM 读取、SMEM 填充、同步和复用；移动端可用；无控制台错误。

- [ ] `LEGACY-002` 教程步骤 4：Vectorized Memory Access
  - 当前状态：根入口只有大纲，没有详细教程和独立交互演示。
  - 需要交付：`float4`/128-bit 载入、地址对齐、尾部处理和标量访问对比讲解。
  - 演示建议路径：`demos/step4-vectorized-memory-access.html`
  - 验收条件：可切换标量与向量访问；展示指令数、事务利用率、对齐与非对齐结果；移动端可用；无控制台错误。

- [ ] `LEGACY-003` 教程步骤 5：Warp Tiling & CUTLASS Hierarchy
  - 当前状态：根入口只有大纲；现有 `cutlass-hierarchy/` 是来源专题，不能替代教程步骤 5 的完整讲解和独立演示。
  - 需要交付：Threadblock Tile、Warp Tile、Thread Tile、寄存器累加器及层级数据复用讲解，并与步骤 3、4 串联。
  - 演示建议路径：`demos/step5-warp-tiling-cutlass.html`
  - 验收条件：可下钻各层 tile；参数变化能反映线程映射、寄存器/SMEM 压力和复用关系；移动端可用；无控制台错误。

## 已完成

当前无从本清单关闭的事项。

## 维护规则

1. 每次交付前搜索 `下一轮`、`未完成`、`TODO`、`FIXME` 和评审残余问题。
2. 未在本次交付完成的事项必须先登记，再提交。
3. 关闭事项时，将 `[ ]` 改为 `[x]`，写明完成提交、测试方法和结果。
4. 来源专题与教程步骤分别验收；内容相关不能视为同一交付物。

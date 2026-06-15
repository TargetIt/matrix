# 遗留问题

本文件记录尚未完成、暂缓处理或需要后续验证的事项。完成一项后必须勾选，
并补充完成提交和验证结果；新增需求如果未在当次交付中闭环，也必须登记在此。

## 当前未完成

当前无未完成事项。

## 已完成

- [x] `LEGACY-001` 教程步骤 3：Shared Memory Caching & Block Tiling
  - 完成内容：协作加载、加载后同步、SMEM 计算、计算后同步、下一 K 轮/写回五阶段。
  - 边界范围：M/N 输出边界、K-tail 和 SMEM 补零槽；block 坐标保持 tile 网格对齐。
  - 验证结果：300 个参数/阶段状态通过；桌面与 390px 元素包围盒无越界；无运行错误。
  - 完成提交：`fix: address CUDA tutorial review findings`。

- [x] `LEGACY-002` 教程步骤 4：Vectorized Memory Access
  - 完成内容：float/float2/float4、对齐前导、向量主体、标量尾部及 128B segment 利用率。
  - 修复内容：移除 128 元素绘制上限，最大 `offset=7,count=128` 时完整显示到索引 136。
  - 验证结果：5,808 个参数组合通过；可视元素数量覆盖完整请求区间；390px 无越界。
  - 完成提交：`fix: address CUDA tutorial review findings`。

- [x] `LEGACY-003` 教程步骤 5：Warp Tiling & CUTLASS Hierarchy
  - 完成内容：记录选中 warp，并按 4×8 lane 网格显示 32 个 thread accumulator tile。
  - 公式范围：分别报告 A/B 主循环强度和包含最终 C 读写的完整 tile 强度；增加寄存器、active blocks 和 occupancy 估算。
  - 可访问性：SVG 节点具备键盘事件和高对比 `:focus-visible`，移动端提供 44px 全宽文本层级按钮。
  - 验证结果：144 个 policy 组合通过；主循环强度不随 BK 错误变化；390px 无越界。
  - 完成提交：`fix: address CUDA tutorial review findings`。

- [x] `LEGACY-004` 独立 Fail 复审
  - 根页示意百分比已替换为 `sgemm-practice/original.md` 中 RTX 3090 实测数据。
  - 390px 使用 `scrollWidth/clientWidth` 与全部可见元素包围盒双重验证，三个新增演示均无越界。
  - Fail 报告中的移动端溢出未能复现；其余七项意见均已采纳修复。
  - 处理记录：`review/20260614110501-independent.md`。

## 维护规则

1. 每次交付前搜索 `下一轮`、`未完成`、`TODO`、`FIXME` 和评审残余问题。
2. 未在本次交付完成的事项必须先登记，再提交。
3. 关闭事项时，将 `[ ]` 改为 `[x]`，写明完成提交、测试方法和结果。
4. 来源专题与教程步骤分别验收；内容相关不能视为同一交付物。

# 遗留问题

本文件记录尚未完成、暂缓处理或需要后续验证的事项。完成一项后必须勾选，
并补充完成提交和验证结果；新增需求如果未在当次交付中闭环，也必须登记在此。

## 当前未完成

当前无未完成事项。

## 已完成

- [x] `LEGACY-001` 教程步骤 3：Shared Memory Caching & Block Tiling
  - 完成内容：补充详细教程；新增 `demos/step3-shared-memory-block-tiling.html`。
  - 覆盖范围：协作加载、同步屏障、SMEM 复用、写回和边界 tile。
  - 验证结果：18 组矩阵规模/tile/位置组合通过；桌面与 390px 无溢出；无控制台错误；键盘阶段推进通过。
  - 完成提交：`feat: complete CUDA tutorial steps 3-5`。

- [x] `LEGACY-002` 教程步骤 4：Vectorized Memory Access
  - 完成内容：补充详细教程；新增 `demos/step4-vectorized-memory-access.html`。
  - 覆盖范围：float/float2/float4、对齐前导、向量主体、标量尾部及 128B segment 利用率。
  - 验证结果：288 组宽度/偏移/长度/基地址组合通过；桌面与 390px 无溢出；无控制台错误；键盘调参通过。
  - 完成提交：`feat: complete CUDA tutorial steps 3-5`。

- [x] `LEGACY-003` 教程步骤 5：Warp Tiling & CUTLASS Hierarchy
  - 完成内容：补充详细教程；新增 `demos/step5-warp-tiling-cutlass.html`。
  - 覆盖范围：Threadblock/Warp/Thread 层级、线程映射、accumulator、SMEM、算术强度、单双缓冲和资源风险。
  - 验证结果：48 组 policy 参数组合通过；桌面与 390px 无溢出；无控制台错误；层级键盘下钻通过。
  - 完成提交：`feat: complete CUDA tutorial steps 3-5`。

## 维护规则

1. 每次交付前搜索 `下一轮`、`未完成`、`TODO`、`FIXME` 和评审残余问题。
2. 未在本次交付完成的事项必须先登记，再提交。
3. 关闭事项时，将 `[ ]` 改为 `[x]`，写明完成提交、测试方法和结果。
4. 来源专题与教程步骤分别验收；内容相关不能视为同一交付物。

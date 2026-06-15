# 验证规范

## 自动回归

在仓库根目录执行：

```bash
./scripts/test.sh
```

当前自动检查包括：

- GEMM policy 的 block/warp/thread 整除关系。
- 每个 warp 恰好映射 32 个 thread tile。
- threads/block、SMEM 和寄存器资源边界。
- A/B 主循环强度不随 BK 错误变化。
- 完整强度只在完整 K 范围计入 C 读写。
- 来源文件 SHA-256 完整性。
- 页面不再包含人工复用百分比和旧错误公式。
- Step 4 明确区分地址 segment 模型和真实硬件事务。

## 浏览器回归

每次交付至少检查根入口、三个来源专题和六个教程页面：

1. 桌面视口 `1440x900`。
2. 移动视口 `390x844`。
3. 页面无 JavaScript 错误、资源 404、横向溢出或文字遮挡。
4. 所有按钮、选择框、滑块和可点击 SVG 节点可操作。
5. CUTLASS 默认值必须显示 128 threads、12.8 FLOP/B 主循环强度。

## 性能与学习效果

以下项目需要真实浏览器或目标用户参与，不能仅凭代码审查关闭：

- 首次内容绘制小于 2 秒。
- 连续操作 20 轮无崩溃和明显内存增长。
- 动画保持流畅，并尊重 `prefers-reduced-motion`。
- 3 至 5 名目标用户完成 15 分钟探索和概念问答。
- 至少 80% 用户能在 1 分钟内完成指定下钻任务。

没有测试证据时，必须在遗留问题中保持未完成状态。

## 2026-06-15 执行记录

- `./scripts/test.sh`：通过。
- 10 个页面 HTTP 访问：全部返回 200。
- Headless Chrome `1440x900`：10 个页面无 JavaScript 异常、无页面级横向溢出。
- Headless Chrome `390x844`：10 个页面无 JavaScript 异常、无页面级横向溢出。
- CUTLASS 默认值：32 outputs/thread、4 warps / 128 threads、12.8 FLOP/B、
  12.2 FLOP/B、75% occupancy。
- 未执行 FCP、帧率、长期内存和真实用户学习效果测试；继续记录在遗留问题。

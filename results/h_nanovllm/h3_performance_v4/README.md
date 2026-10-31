# H3 性能 v4：数学预算通过，三参考联合门槛仍失败

2026-10-24。三条路径统一关闭 BF16 GEMM 中间累积降精度许可，权重、输入和输出
仍为 BF16；同时比较三组新参考。默认策略此前的数值失败均保留。

本轮相对 FP32 数学参考的全部比较通过，batched BF16 参考也全部通过。但
batch_short 相对逐请求 BF16 参考的 native / paged eager / paged Graph 最大
relative RMSE 分别为 0.541808% / 0.570047% / 0.570047%，超过原 0.5%。
因此 [gate.json](gate.json)仍为 **fail**，pilot 0/6、正式进程 0/36。

这说明对一个数学基准各自满足的误差预算，不能当成 BF16 舍入路径互比的传递性保证。
本轮不改写为 pass，不选择不同数据窗口或放宽阈值。

[v5](../../../docs/h3_performance_v5_protocol.md)将 v3 的既定 FP32 数学预算与
本轮共同累积策略组合，独立重建 gate 和计时；其结果必须同时披露本轮 BF16 互比限制。

证据：[gate.json](gate.json)、[run.json](run.json)、三组 `reference*.json`、
`qualification/` 和 `source_snapshot/`。

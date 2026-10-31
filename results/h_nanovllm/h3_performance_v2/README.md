# H3 性能 v2：逐请求 BF16 参考仍未整体通过

2026-10-24。[v2 协议](../../../docs/h3_performance_v2_protocol.md)采用原 H3 的逐请求
BF16 full recompute 参考，保持 v1 的模型、输入、目标执行路径和所有阈值。

新 gate 在 page_growth（B=8、prompt=255、output=32）的原生连续 KV 路径失败：
最大 relative RMSE 为 **0.524697%**，超过 0.5%；最大绝对误差 0.125。
其他数值项通过，864 个 eager/Graph 向量全部 exact。pilot 0/6，正式计时 0/36。

本轮失败、参考、三侧 logits、日志和源码快照保留，不放宽阈值或改写 gate。
随后 [v3 协议](../../../docs/h3_performance_v3_protocol.md)引入同值 BF16 参数的
FP32 数学参考，独立验收；v3 的结果不覆盖本轮失败。

证据：[gate.json](gate.json)、[run.json](run.json)、`qualification/`、`source_snapshot/`。

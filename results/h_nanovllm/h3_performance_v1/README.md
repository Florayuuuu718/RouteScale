# H3 性能 v1：数值门槛未通过

2026-10-24。模型接入与数据准备已执行，但新数值 gate 为 **fail**；pilot 0/6、正式计时
0/36。保留原始参考、三条路径的全步 logits、失败 gate、运行日志和冻结源码，未放宽门槛。

batch_short（B=8、prompt=128、output=64）的分页 eager/Graph 相对 batched BF16
full recompute 最大 relative RMSE 为 0.556876%，超过 0.5%；最大绝对误差 0.0625。
其余三 case 通过。全部四 case 的 864 个 eager/Graph 向量 bitwise exact。

[后续诊断](reference_shape_diagnostic.json)仅将参考改成逐请求 full recompute，三条
路径在 batch_short 都通过相同门槛；batch=8 full recompute 本身与逐请求参考也存在
0.429419% 的差异。选中失败步的第一层探针首次在 MLP c_proj 分叉，此前 QKV、
attention 输入／输出投影和 c_fc 均 exact。这支持参考 GEMM 行形状相关的 BF16 舍入。
不把这一诊断表述为运行时 bug 已修复。

[v2 独立协议](../../../docs/h3_performance_v2_protocol.md)恢复原 H3 使用的逐请求参考，
新建配置、参考、资格验证、pilot 和计时结果；本目录失败不会改写为 pass。

关键证据：[gate.json](gate.json)、[run.json](run.json)、
[reference_shape_diagnostic.json](reference_shape_diagnostic.json)、`qualification/`、`source_snapshot/`。

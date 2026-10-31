# H3 性能 v3：FP32 数学参考确认原策略仍有超预算点

2026-10-24。参考使用已经 BF16 舍入的参数值提升为 FP32、逐请求 math SDPA，
关闭 TF32；三条实际目标路径仍为 BF16。

native/page_growth 最大 relative RMSE 为 0.530423%，paged/batch_short 为
0.514894%，均超过未修改的 0.5%。[gate](gate.json)为 fail，pilot 0/6，正式计时 0/36。
保留参考、三侧 logits、失败与源码快照，不以更换参考掩盖误差。

[随后单开关诊断](reduction_policy_diagnostic.json)统一关闭 BF16 GEMM 中间累积
降精度许可，在相同 BF16 权重、输入和输出 dtype 下使两个失败场景的误差降为
0.334612% / 0.430108%。第一层 MLP 投影固定输入控制也显著减少 batch/slice 差异。
这是局部诊断，不是本轮 gate 通过。

[v4 协议](../../../docs/h3_performance_v4_protocol.md)将此数值策略统一应用于所有
目标路径，并同时要求逐请求 BF16、批量 BF16、FP32 数学参考全部通过，独立重做
资格验证、pilot 与正式矩阵。本目录失败保持不变。

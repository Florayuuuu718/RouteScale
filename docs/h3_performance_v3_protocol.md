# H3 性能 v3：同值 BF16 参数的 FP32 数学参考

2026-10-24，在任何 v3 参考、资格验收或计时前冻结。
配置：[h3_performance_v3.json](../config/serving/h3_performance_v3.json)。
继承 [v1 协议](h3_performance_protocol.md)的全部目标执行路径、工作量、两种计时范围、
pilot、正式矩阵、固定池、统计方法和数值阈值；目标路径不加任何新 kernel/线性层策略。

## 为什么建立新的数值基准

- v1：batched BF16 full recompute 下，batch_short 分页路径最大 relative RMSE
  0.556876%，超过 0.5%。[失败](../results/h_nanovllm/h3_performance_v1/gate.json)保留，正式进程 0/36。
- v2：沿用原 H3 的逐请求 BF16 full recompute，batch_short 已通过，但 page_growth
  的 native cached 最大 relative RMSE 为 0.524697%，超过 0.5%。
  [失败](../results/h_nanovllm/h3_performance_v2/gate.json)保留，正式进程 0/36。
- [v1 诊断](../results/h_nanovllm/h3_performance_v1/reference_shape_diagnostic.json)发现，
  只改变原生 full recompute 的 batch 形状就会出现差异；第一层探针先在 MLP c_proj
  输出分叉。两种 BF16 参考不能凭其中一种通过而视为通用等价证明。

v3 固定更高精度的数学参考，不再根据目标路径挑选 BF16 参考批次形状：

1. 加载同一 artifact，先把参数完整舍入为 BF16，计算与三条目标路径相同的 BF16
   state-dict 哈希；随后将这些**已经舍入的参数**提升为 FP32。不会使用原 checkpoint
   中被 BF16 舍弃的低位，也不重训练或重导出。
2. 逐请求 full recompute，所有运算 FP32；强制 SDPA math backend、关闭 TF32。
   保存 FP32 的全步全词表 logits，不再将参考输出舍入回 BF16。
3. 三条实际被测路径仍是原来的 BF16 权重、BF16 计算及各自 attention backend；
   逐向量比较目标 `.float()` 与 FP32 数学参考。沿用 `atol=0.125, rtol=0.02`、
   relative RMSE≤0.005、finite，以及 eager/Graph bitwise exact 要求，不放宽门槛。
4. 新建参考、三进程资格验证、2592 个 BF16 目标向量、864 个 Graph exact 向量，
   全部通过后才能进入六个 pilot 与 36 个正式进程。FP32 参考耗时绝不进入性能表。

输入 seed、全部请求与后续 token、输出长度、目标模型、KV 分配和计时边界保持不变。
若这一数学参考仍不通过，就保留失败并继续定位模型／数值实现，不允许正式计时。

v3 的 pass 只证明各目标相对这一数学基准满足预算；不改写 v1/v2 的失败，也不推出
任意两个 BF16 批次形状之间都能满足 0.5% 阈值。性能表必须写明此参考口径及回放限制。

## 产物与执行

结果放 `results/h_nanovllm/h3_performance_v3/`；请求、大张量、日志、缓存分别使用
`runs/workloads/h3_performance_v3/`、`runs/validation/h3_performance_v3/`、
`runs/logs/h_nanovllm/h3_performance_v3/`、`runs/cache/h3_performance_v3/`。
冻结源码、配置与协议保存在结果目录的 `source_snapshot/`。

```bash
.venv/bin/python scripts/run_h3_performance.py --config config/serving/h3_performance_v3.json
runs/envs/nano-vllm/bin/python scripts/audit_h3_performance.py --config config/serving/h3_performance_v3.json --check-only
```

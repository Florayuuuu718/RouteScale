# H3 性能 v2：沿用原 H3 的逐请求 BF16 参考

2026-10-24，在任何 v2 GPU 验收或计时之前冻结。
配置：[h3_performance_v2.json](../config/serving/h3_performance_v2.json)。
继承 [v1 协议](h3_performance_protocol.md)的路径、输入生成、两种计时边界、矩阵、
预热、三次独立进程、pilot CV 门槛、64 MiB 分页池、资源检查与所有数值阈值。

## 参考口径变更及原因

v1 的 batched full-recompute 参考下，batch_short 的分页 eager/Graph 最大 relative
RMSE 为 0.556876%，超过 0.5%；其余 case 通过，864 个 eager/Graph 向量全都 exact。
v1 已保留失败 gate，pilot 和正式进程均为零，不能被 v2 的 pass 覆盖。

[独立参考形状诊断](../results/h_nanovllm/h3_performance_v1/reference_shape_diagnostic.json)
保持权重、请求、token、精度和模型代码不变，只比较 batch=8 与八次 batch=1 的
full recompute。选中的首个失败步里，第一层 ln_1、QKV、attention 输出／投影、
ln_2、MLP c_fc 均 exact，首个非 exact 探针为 MLP c_proj。证据支持参考 GEMM
行形状引起的 BF16 舍入差异，不是已经修复了 attention/KV 或 Graph bug。

原 H3 的 `verify_h3_routescale.execute_wave` 与独立审计一直逐请求运行原生 BF16
full recompute，再拼接 logits。0.5% 门槛沿用自这一合同。v2 明确采用同一参考口径：
每请求 `model(tokens[None])[0][:,-1]`，然后按请求顺序拼接；目标路径继续实际 batch=8。
不改变权重、模型计算、输入窗口、输出步数、KV 策略或 attention backend。

诊断中 batch_short 的分页相对逐请求参考最大 relative RMSE 为 0.467827%，
native cached 为 0.464234%，batch=8 full recompute 本身为 0.429419%。
这些只是 v1 输入上的诊断，不作为 v2 通过依据。v2 必须在新目录重建全部四 case
逐请求参考、三条目标路径、2592 个数值向量与 864 个 Graph exact 向量，再做新 pilot。

v2 的性能结果只在明确的逐请求参考合同下成立；继续披露 v1 的 batched-reference
失败。不能宣称所有 BF16 batch 形状都相互满足 0.5% 阈值。

## 执行与产物

- 结果：`results/h_nanovllm/h3_performance_v2/`。
- 请求：`runs/workloads/h3_performance_v2/requests.json`；token 数组与 v1 的生成方式和 seed 完全相同。
- 大张量：`runs/validation/h3_performance_v2/`；日志与缓存使用各自 v2 目录。
- 启动前冻结的源码／协议副本保存在结果目录的 `source_snapshot/`。

```bash
.venv/bin/python scripts/run_h3_performance.py --config config/serving/h3_performance_v2.json
runs/envs/nano-vllm/bin/python scripts/audit_h3_performance.py --config config/serving/h3_performance_v2.json --check-only
```

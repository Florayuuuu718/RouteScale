# H3 性能 v4：统一 BF16 GEMM 累积策略，三参考同时验收

2026-10-24，在任何 v4 参考、资格验证或计时前冻结。
配置：[h3_performance_v4.json](../config/serving/h3_performance_v4.json)。
继承 [v1](h3_performance_protocol.md)的全部四 case、固定输入、三条目标路径、两种计时
范围、64 MiB 分页池、统计方法、六个 pilot 与 36 个正式进程。模型权重、输入／输出
仍为 BF16，关闭 autocast，未改模型结构、attention backend、软件依赖或上游 checkout。

## 新的共同数值策略

v1/v2 使用两种 BF16 参考都分别有失败；[v3](../results/h_nanovllm/h3_performance_v3/gate.json)
采用同值 BF16 参数的 FP32 数学参考后，native/page_growth 和 paged/batch_short
仍分别达到 0.530423% / 0.514894%，超过 0.5%。这三轮失败全部保留，正式进程均为零。

本环境 `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` 默认 True。
[PyTorch 2.8 官方说明](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)
说明该许可允许部分 GEMM 中间累积提前降精度，提供 False 开关。它不是把输入、
权重或结果改为 FP32，也不保证所有 BF16 计算都逐位一致。

[单开关诊断](../results/h_nanovllm/h3_performance_v3/reduction_policy_diagnostic.json)中，
用完全相同的第一层 MLP c_proj 输入和 BF16 权重，batch/slice 不同元素数由 210339
降至 422，最大差由 0.0078125 降至 0.001953125；两个失败场景相对原 v3 FP32 参考
的最大 relative RMSE 分别降为 0.334612% / 0.430108%。这是有限范围的因果控制，
不提前宣称全部 workload 或 Graph 已通过。

v4 三条目标路径及两种 BF16 full-recompute 参考统一设为：

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

设置在模型初始化、warmup、Graph capture 和任何正式计算之前，实际值写入 provenance。
三侧保持同一策略；结果属于这一显式策略，不冒充库默认配置下的性能。

## 比原先更严格的资格门槛

新进程分别建立三组完整参考，并要求每条目标路径**同时**通过全部参考：

1. 原 H3 逐请求 BF16 full recompute，为主参考；
2. batch=实际 cohort 的 BF16 full recompute，覆盖 v1 的参考形状；
3. BF16 舍入后参数提升为 FP32 的逐请求 math SDPA 参考，关闭 TF32，覆盖 v3 的数学基准。

全部使用相同 token 数组与训练参数值。每参考 864 个向量，三条 BF16 目标路径合计
2592 个不同向量、7776 次目标—参考比较；864 个 eager/Graph 向量还须 bitwise exact。
finite、pointwise `atol=0.125, rtol=0.02`、每向量 relative RMSE≤0.005 均保持原值。
没有按路径选择有利参考，没有排除失败 case，也没有把 FP32 参考时间加入性能表。

随后按原规则执行六个独立 pilot，CV≤15%、释放／输出／页增长／显存余量全部通过后，
再运行四 case×三路径×三独立进程共 36 个正式进程。原始失败不得覆盖；无收益可作为
有效结果，不能根据速度挑选形状。固定回放不是自然生成、上游公共 API 或在线 SLO。

## 执行与路径

结果、请求、大张量、日志、编译缓存分别使用 `h3_performance_v4` 独立目录。
冻结源码、配置和协议放结果目录 `source_snapshot/`。v1/v2/v3 JSON 在本轮开始前
记录哈希，结束及独立审计时重新核对。

```bash
.venv/bin/python scripts/run_h3_performance.py --config config/serving/h3_performance_v4.json
runs/envs/nano-vllm/bin/python scripts/audit_h3_performance.py --config config/serving/h3_performance_v4.json --check-only
```

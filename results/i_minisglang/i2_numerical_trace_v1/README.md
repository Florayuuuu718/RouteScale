# I2 两个相继数值分叉的定位

本轮保留原 I2 v1 的 HF FP32 失败；以下均为诊断，`performance_allowed=false`。
工程说明、因果图和复现入口见 [数值定位说明](../../../docs/i2_numerical_findings.md)。

| 证据 | 结果 |
|---|---|
| [原生逐层追踪](trace001/audit.json) | 两侧 cold/warm 的256个输出向量分别 exact 复现未追踪运行；28层前缀K/V全部一致，首个差异是 layer0 attention |
| [固定 Q/K/V 干预](trace001/probe001/audit.json) | full512 不拆KV，suffix128 自动拆为171；共同 no-split 或固定128分块均使局部输出 exact；CPU FP64 独立参考核对通过 |
| [单独 no-split](cache_nosplit001/comparison.json) | cold64向量 exact，warm64仍失败，最大相对RMSE .040045；失败保留 |
| [no-split 再追踪](trace_nosplit001/audit.json) | 前序算子全部 exact，首个差异移至 layer0 MLP down_proj，251元素、max abs=.001953125 |
| [同输入 down_proj](trace_nosplit001/downproj001/result.json) | 512/128行 GEMM 分别 exact 复现追踪；共同512行使输出 exact |
| [两项共同控制](cache_shape001/comparison.json) | 原 cold/warm 全部128个全词表向量 exact |

固定模型、精度、原 token ID、资源池和原生树；共同策略另放在两个明确的 adapter，
不藏入观察器。linear tile512 增加真实补齐运算，不能把它的性能称作未改变计算形状的上游原生性能。
张量保存在忽略的 `runs/validation/i2_numerical_trace_v1`，Git 保存哈希、源码快照和小型指标。

后续带新seed/768-token输入、七配置的完整资格验证和HTTP复验见
[资格验证记录](../i2_shape_qualification_v1/README.md)。正式计时使用独立冻结范围，
不会追认原HF FP32全模型门槛。诊断包可用以下命令重新核对：

```bash
runs/envs/mini-sglang/bin/python scripts/recheck_i2_shape.py --kind findings
```

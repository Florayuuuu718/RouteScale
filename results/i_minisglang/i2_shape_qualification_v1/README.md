# I2 共同形状控制策略：数值与 HTTP 资格验证

**本资格范围通过，不直接签发性能 gate。** TP1、Qwen3-0.6B BF16、RTX5060 Laptop、
FlashInfer fa2、共享 tokenizer；共同应用 prefill no-split 与四种 linear 的 tile512。
原HF FP32全模型门槛仍失败。此处验证同一原生 BF16 引擎内的策略等价性。

- [qual001 独立数值审计](qual001/audit.json)：独立 Naive 参考，再用七个新进程重放全部输入。
  28请求、1792个参考向量、12544个目标向量，全部逐元素 exact。
- 原 seed20261031、prompt512、batch1/3、prefix0/384；另加未参与定位的
  seed20261101、prompt768、batch1/2、prefix0/512。每请求64步、全词表151936维。
- 两种 chunk、两种实际 Graph bucket 和真实异步 Overlap 均覆盖；冷KV填NaN，
  命中、输出预算、释放及物理页/table唯一性均检查。
- [lifecycle001 独立HTTP审计](lifecycle001/independent_audit.json)：七服务、84请求、280项检查通过。
  EOS/length、混合长短、decode断开、实际排队取消、deadline、后续probe和有界关闭均覆盖。
  每个 engine 的事件均确认实际安装 policy 和两份源码哈希。
- [新增CPU检查](cpu_tests.json)：8项数学参考/adapter检查、4项计时准入负对照通过。
  原I2 v1的12项检查和失败证据另存，未重写。

`exact` 指 `torch.equal` 的元素值完全一致。实际参考范围以 provenance 的
`reference_scope=within_native_BF16_strategy_invariance` 为准；嵌套原配置保留原容差来源，
并未将其旧HF参考标签当作本次实际参考。

三个验收边界独立：原HF FP32等价、当前引擎内部策略等价、真实HTTP生命周期。
资格验证后的正式性能另见 [形状控制计时协议](../../../docs/i2_shape_performance_protocol.md)。
本记录不提供在线SLO、自然生成质量、TP2/4或其他attention后端的验收。

```bash
runs/envs/mini-sglang/bin/python scripts/qualify_i2_shape_policy.py --attempt fresh_qual001
runs/envs/mini-sglang/bin/python scripts/audit_i2_shape_qualification.py \
  --directory results/i_minisglang/i2_shape_qualification_v1/fresh_qual001
runs/envs/mini-sglang/bin/python scripts/check_i2_shape_lifecycle.py \
  --qualification results/i_minisglang/i2_shape_qualification_v1/fresh_qual001 --attempt fresh_lifecycle001
runs/envs/mini-sglang/bin/python scripts/audit_i2_shape_lifecycle.py \
  --directory results/i_minisglang/i2_shape_qualification_v1/fresh_lifecycle001
```

完整张量、环境、模型与原始日志留在忽略的 `runs/`。已有目录拒绝覆盖；协议、失败、
中间干预和source snapshot均保留。工程根因见 [数值定位说明](../../../docs/i2_numerical_findings.md)。

# I2 外部 HF FP32 对齐：舍入定位与提高精度验证

原门槛没有放宽：`atol=.125, rtol=.02, relative_RMSE<=.005`，全部64步、全部151936词表
元素都须有限且通过。相同 Qwen3-0.6B BF16 checkpoint、原始 batch1/3、512-token、
zero/shared、cold/warm、固定 continuation，共1024向量。未开展性能计时或使用付费资源。

原 BF16 结果仍失败；本轮验证提高运算精度能否满足同一外部参考门槛，不修改历史结论。
**FP16主路径 + FP32残差/层间与最终norm** 在 Naive/Radix 各1024向量、合计2048向量
全部通过原阈值，最坏相对RMSE=.00322175（0.322%）。汇总见 [summary.json](summary.json)。

## 为什么失败仍保留

原参考把 BF16 权重升为 FP32 运算，目标是 BF16 中间激活、KV 和融合算子。相同权重
来源不意味着相同舍入边界。上一轮 shape policy 验证的是引擎内部策略一致性；本轮
独立比较外部 HF 参考。两项结果均保留，不以新参考追认旧通过。

在新的完整批次复现中，原 Mini BF16 的全部1024向量与原运行逐元素一致；HF FP32
也逐元素复现全部1024个旧参考。HF 自身 eager BF16 对该 FP32 参考同样0/1024通过，
最大相对 RMSE 为5.6175%。因此原0.5%门槛并非标准 HF BF16 在本 workload 已满足的预算。
这支持精度路径是重要因素，但不能单凭此判断所有 Mini 算子无 bug。

## 首个外部分叉已复现

逐层记录首个 cold prefill 的末128行、28层、423个算子输出。Embedding exact，首个
非 exact 为 layer0 input RMSNorm：Mini BF16 对 HF FP32 相对 L2=.00165430；
Mini 对 HF BF16 为 .00285339。分叉在 attention 和缓存之前。

同一实际 embedding 输入、同一 checkpoint norm weight 上，重新运行 FlashInfer RMSNorm、
HF BF16 公式与 HF FP32 公式，分别 exact 复现三侧已记录的 norm 输出。独立 CPU FP32
公式也通过检查。HF BF16 在乘 norm weight 前后各有低精度舍入；FlashInfer 将归一化与
乘权重融合后舍入。显式单次舍入 PyTorch 数学式与 FlashInfer 只差2/131072元素，局部
相对 L2=1.42e-5，提示另有很小的归约/浮点边界差异。

仅把残差与 norm 联合改为 FP32 的早期16向量诊断仍失败，最大相对 RMSE=.01923。
不能将全模型误差唯一归给第一个 norm，也不能将一次局部对齐等同于完整模型修复。

## 完整数值结果

| 路径 | 缓存 | 通过向量 | 最大相对 RMSE | 最大绝对误差 | top-1相同 |
|---|---|---:|---:|---:|---:|
| 原 Mini BF16，完整复现 | Naive | 0/1024 | .04015170 | .5203762 | 1001/1024 |
| HF eager BF16，完整对照 | 独立连续 KV | 0/1024 | .05617524 | .9247770 | 984/1024 |
| 原生 Mini/FlashInfer FP16 | Naive | 1024/1024 | .00466755 | .0785446 | 1019/1024 |
| 原生 Mini/FlashInfer FP16 | Radix | 1023/1024 | .00526206 | .0785446 | 1018/1024 |
| Mini FP32 paged math | Naive | 1024/1024 | .000009430 | .00014925 | 1024/1024 |
| Mini FP32 paged math | Radix | 1024/1024 | .000009430 | .00014925 | 1024/1024 |
| FP16主路径 + FP32残差/norm | Naive | 1024/1024 | .00322175 | .1176071 | 1020/1024 |
| FP16主路径 + FP32残差/norm | Radix | 1024/1024 | .00322175 | .1176071 | 1020/1024 |

纯FP16在 `b1_shared_warm_0` 第3步（从0计）相对RMSE=.00526206，虽然逐元素容差
无违规、top-1相同，仍严格判失败。后续候选采用FP16主算子和KV、FP32残差及层间/最终norm，
不继续沿用“纯FP16已满足完整缓存范围”的结论。该联合方案使用显式PyTorch norm，
其额外kernel发射与运行成本尚未测量。

FP16 修改权重格式、激活与 KV 精度，保持同一个 BF16 checkpoint 来源；它没有恢复原权重
丢弃的位。BF16 checkpoint 中120378/751632384个序列化元素转换 FP16 后数值改变，
最大变化2.98e-8，无非有限值；该元素总数按 checkpoint 张量计数。
FP16 动态范围较小，本轮有限性通过不能替代更长上下文或新输入的验证。

FP32 paged math 保留 Mini 模型结构、合并投影、原调度器、真实页表与缓存分配，
权重由 BF16 升 FP32；norm、RoPE、激活和真实页上的 attention 使用 PyTorch FP32。
它绕过 FlashInfer attention 执行，验证所覆盖的模型/调度/缓存语义，不证明原 BF16
FlashInfer 算子逐个正确。FP32 输出也不要求 bitwise exact。

FP16 的数值余量较小，且 Naive 仍有5/1024步 top-1 不同。原协议只报告 top-1，
不以其替代全词表门槛；这些固定 continuation 结果不保证自然生成序列或质量一致。

## 证据与边界

- [Naive 精度矩阵及独立审计](full_naive001/audit.json)：3259项检查。
- [完整旧结果复现与同输入 norm 探针](full_reproduction001/audit.json)：3256项检查。
- [Radix 完整精度矩阵](full_radix001/audit.json)：2219项检查，包含真实384-token暖缓存命中。
- [混合精度 Naive](mixed_naive001/audit.json)与 [Radix](mixed_radix001/audit.json)：各1124项检查，完整通过。
- [CPU测试](cpu_tests_precision_candidate.json)：7项，覆盖 GQA、物理页重排、suffix因果mask、混合请求、BF16/FP16下FP32残差与恢复、Graph拒绝。早期6项记录另存。
- [审计负对照](audit_negative_controls.json)：阈值、记录误差、源码三种改动均被拒绝。
- [逐层精度对照](cross_precision_layers.json)及 [执行协议](../../../docs/i2_external_alignment_protocol.md)。

`pilot001` 的 FP16 构造因冻结 dataclass 修改错误而中止；新轮使用 `dataclasses.replace`。
`pilot002` 将 batch3 缩为单请求，整体原批次复现检查失败；其小样本结果不计完整验收。
两轮源码、进程账本与 `validation_issue.json` 保留。正式结论来自新的完整原始批次。

本轮不切换生产/HTTP默认精度，不签发七配置或性能 gate。新精度下 Graph/Overlap、
新输入、混合长度、HTTP生命周期及吞吐成本尚待验证。原BF16失败和旧范围0/21保持不变；
先冻结候选精度与验收范围，再进入 I3。既有 BF16 shape-controlled 性能结果仍按其原范围解释。

复现使用新 attempt，避免覆写冻结证据：

```bash
runs/envs/mini-sglang/bin/python scripts/diagnose_i2_external_alignment.py \
  --attempt fresh_full --scope full --modes hf_fp32 hf_bf16 native_bf16 native_fp16 fp16_residual_fp32 fp32_math
runs/envs/mini-sglang/bin/python scripts/audit_i2_external_alignment.py \
  --directory results/i_minisglang/i2_external_alignment_v1/fresh_full
```

逐层观察、范式干预和代码版本都在各轮 source snapshot；完整张量在忽略的 `runs/validation`。
工程定位记录在仓库，个人复盘和面试材料仍留待全部实验完成后统一整理。

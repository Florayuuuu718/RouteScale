# I2 单卡消融 v1：数值门槛未通过

后续外部HF逐层对齐与提高精度方案见[独立验证记录](../i2_external_alignment_v1/README.md)。
本目录原BF16门槛失败和0/21正式进程保持不变。

2026-10-31，本机 RTX 5060 Laptop 8 GiB、Qwen3-0.6B BF16、TP1、共享 tokenizer。
**I2 尚未完成，正式性能进程 0/21，无吞吐倍率或在线 SLO 结论。**
没有启动付费资源。I1 的四次验收、原上游和历史失败保持不变。

## 已完成与未通过

| 工作 | 真实结果 | 证据 |
|---|---|---|
| Overlap 生命周期扩展 | EOS/abort 等待在途 GPU 工作后再释放；屏蔽迟到结果；长度按 host token 数判断；退休 UID 有界 | [两个补丁](../../../patches/minisgl-i2/series.json) |
| 显式 Graph buckets | `[1,2,4]` 与 `[1,3]` 可分别配置，不依赖上游最大值推导 | 同上 |
| CPU 检查 | 12 项通过，包含在途取消、长度竞争、退休集合、页/table 索引、完整 logits 和独立审计负对照 | [cpu_tests.json](cpu_tests.json) |
| 七配置 HTTP 生命周期 | 84 请求、280 项检查通过；正常关闭 2.94–4.50 秒，无强杀/升级/进程或 GPU 残留 | [独立审计](lifecycle_diag001/independent_audit.json) |
| HF FP32 数学参考 | 独立进程生成全部 1024 个全词表参考向量 | [reference.json](attempt001/reference.json) |
| Naive 完整数值对照 | 1024/1024 向量有限，0/1024 满足冻结门槛；top-1 1001/1024 相同，不能替代数值验收 | [失败 gate](attempt001/numerical_gate.json) |
| 其余六配置正式数值/全部性能矩阵 | 首项失败即停，未执行；后续 Radix 仅用于诊断 | [进程账本](attempt001/processes.json) |

HTTP 矩阵覆盖 Naive、Radix、chunk=256/128、Graph `[1,2,4]`/`[1,3]`、Overlap。
每配置 12 条真实模型请求：EOS、length、混合长短、decode 断开、四个实际入模 blocker、
排队取消、deadline 取消、后续正常请求。检查每请求单一客户端终态、后端停止输出/调度、
frontend/detokenizer/table/KV 活跃引用和关闭预算。该矩阵没有重跑七种配置下的全部
I1 worker 故障组合，也没有完成完整数值等价或 TP2/4 验收。

服务保留真实原生采样，关闭旧 I1 每步同步 finite hook，以免把 Overlap 串行化。
数值回放另用异步 GPU clone，整波结束后复制全 logits；正式计时仍由数值门槛锁住。

## 数值失败与隔离诊断

冻结门槛为 `atol=0.125, rtol=0.02, relative_RMSE<=0.005`，全部步骤、全部词表元素有限且通过。
参考把同一 BF16 checkpoint 升为 FP32 运算；目标保持 BF16，全部关闭 BF16 GEMM
reduced-precision reduction 和 TF32。Naive 最大绝对误差 0.52038、最大相对 RMSE
0.040152，共 291429 个元素越过逐元素容差。独立审计重新打开全部张量确认失败，
并明确标出六个未执行配置。旧 gate 没有改成通过。

| 诊断 | 范围与观察 | 能支持的结论 |
|---|---|---|
| [Naive / 原生 Radix](native_strategy_diagnostic.json) | 同一固定输入的 1024 个向量中，960 个字节完全相同；b1 shared warm 的 64 个未过门槛，最大相对 RMSE 0.031703 | 差异集中在本次单请求暖前缀路径；不是所有 Radix 请求都分叉 |
| [独立 BF16 舍入参考](rounding_diagnostic.json) | 四个 cold 请求各四步，共 16 个向量；标准 HF BF16 与显式融合舍入的 PyTorch 参考均未通过 | 仅匹配已知融合算子的舍入边界还不够，不能据此唯一归因 |
| [孤立算子](kernel_diagnostic.json) | 单请求冷 prefill、28 层；相同原生输入下，attention 相对 RMSE 最大 0.001674，其他 norm/RoPE/激活诊断更小，全部有限 | 孤立算子误差较小；不构成全模型通过，也不能单凭这一项排除所有实现问题 |
| [共同 linear tile512 干预](linear_shape_diagnostic.json) | 两侧同时固定 BF16 prefill linear 行形状；cold 64 向量 exact，warm 仍失败，最大相对 RMSE 0.027723 | 固定 GEMM 行形状没有单独解决该差异；此干预未进入正式配置/性能 |
| [HF 自身精度边界](reference_precision_diagnostic.json) | 同一 HF eager BF16 与 BF16 权重升为 FP32 运算，四个 cold 请求各四步；0/16 通过，最大相对 RMSE 0.034362 | 本门槛也未被 HF 自身 BF16 路径满足；不能把原生服务失败唯一归于缓存 |

这些都是诊断，`performance_allowed=false`。没有使用 FP32 诊断吞吐冒充 BF16 性能，
没有把 top-1 接近、文本合理、孤立算子通过或 HTTP 正常当成全模型数值通过。

后续更新：上述首分叉已在独立新目录定位到自动split-KV和BF16 GEMM行形状，见
[逐层与同输入干预](../i2_numerical_trace_v1/README.md)。两侧共同控制后，另立协议，
使用新seed/长度与全部七配置验证，并在明确的引擎内部策略等价范围完成
[21进程系统消融](../i2_shape_performance_v1/README.md)。本页的HF FP32失败、旧gate和0/21
仍保持原结论；新范围没有追认这轮数值等价或性能。

## 文件与复现

- [冻结协议](../../../docs/i2_single_gpu_protocol.md)、[配置](../../../config/serving/i2_single_gpu.json)。
- `attempt001`：完整 HF 参考、首个 Naive 数值失败及独立审计；snapshot 保留当时源码。
- `pilot001`：更早的参考生成探针，仅有 reference，不是性能 pilot。
- `native_diagnostic001`：原生 Radix 诊断，仍使用同一 HF 参考，数值状态为 fail。
- `lifecycle_diag001`：七个独立服务，单独的生命周期诊断；不签发性能 gate。
- 全量 tensor、原始日志、模型、环境和缓存位于忽略的 `runs/`；Git 保存小型指标、哈希和源码快照。

```bash
runs/envs/mini-sglang/bin/python scripts/prepare_i2_minisglang.py
CUDA_VISIBLE_DEVICES='' runs/envs/mini-sglang/bin/python -m unittest discover -s tests -p test_i2_single_gpu.py -v
# 数值审计返回失败是预期；原 failed gate 的独立审计源码另存 audit_source_snapshot。
runs/envs/mini-sglang/bin/python scripts/audit_i2_progress.py
runs/envs/mini-sglang/bin/python scripts/audit_i2_lifecycle_diagnostic.py \
  --directory results/i_minisglang/i2_single_gpu_v1/lifecycle_diag001 --check-only
# 新运行必须使用新 attempt；当前原生 v1 会在数值首项失败后停下。
runs/envs/mini-sglang/bin/python scripts/run_i2_single_gpu.py --attempt fresh001 --phase all
# 独立检查取消/清理，不放行数值或性能：
runs/envs/mini-sglang/bin/python scripts/run_i2_single_gpu.py --attempt fresh_lifecycle001 --phase lifecycle-diagnostic
```

继续实验仍按本机 I2 → I3 → H3 计时复验和完整交付准备 → 集中多卡的顺序。
Attention backend 对照、TP2/4、在线负载和学习/面试整理均没有在本轮计为完成。

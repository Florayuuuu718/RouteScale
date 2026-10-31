# H2 请求调度：fixed cohort / continuous admission

日期：2026-10-23。**跨调度数值预检未通过，正式性能进程 0/18**。记录见
[`run.json`](run.json) 和 [`preflight.json`](preflight.json)。没有本轮吞吐/TTFT 排名，
不沿用历史 Graph/Prefix 倍率。这里是原策略的历史失败记录；完整模型修正和正式
对照已在独立的 [`h2_scheduling_tile512/`](../h2_scheduling_tile512/README.md) 完成，
不将本目录的 gate 改为通过。

两侧均为 Qwen3-0.6B、BF16、同一修正策略、Graph 开启、Prefix 关闭、8 个在途槽位、
32×256 的相同物理 KV pool。唯一改变的是整组结束才接下一组，还是有空位就补入。
fixed 也立即返回已完成请求并紧凑执行剩余请求，不故意计算 padding。

三个场景分别为等长突发、长短混合突发、长短混合延迟到达；每场景 16 个请求。
数值探针使用逻辑时钟和共同 forced 续写；正式性能使用真实时间到达和上游 sampler，
两者分离。完整请求协议、时延公式、门槛与命令见
[`h2_scheduling_protocol.md`](../../../docs/h2_scheduling_protocol.md)。

## 新数值边界

两个独立 GPU 主预检均执行完整请求预算。既定阈值仍为 atol=0.05、rtol=0.02、
relative RMSE≤0.005；未修改阈值或使用采样文本代替验证。

| 场景 | 每请求每步全词表向量对照 | 精确一致向量 | 最大绝对误差 | 最大 relative RMSE | 结果 |
|---|---:|---:|---:|---:|---|
| equal_burst | 512 | 512 | 0 | 0 | 通过 |
| mixed_burst | 576 | 288 | 0.34375 | 3.974% | 未通过 |
| mixed_staggered | 576 | 288 | 0.34375 | 3.974% | 未通过 |

一共比较 1,664 个完整词表向量。混合场景中，最初 8 个请求全部精确一致，包含经历
补位/批大小变化的旧长请求全部 64 步；差异集中于后补入的 8 个请求。
CPU 状态、在途请求上限、请求结束边界、NaN-poison 有限性及资源释放均通过。
这些证据不能替代全模型数值门槛，也不是模型质量评测。

## 已完成的定位

[`shape_diagnostic.json`](shape_diagnostic.json) 在新 GPU 进程中做 cold Prefill 控制：

- batch=8 A/A 精确一致，独立 cold batch=8 也与保存的原参考精确一致；
- 同样请求拆成 batch=4 即复现差异，不依赖仍在运行的其他请求或其 KV；
- 第一层归一化输入、QKV、旋转后的 Q/K/V、attention 输出均精确一致，**首个分叉
  在 `o_proj` 线性层**，最大绝对误差 0.0078125；
- 关闭 BF16 reduced-precision reduction 开关后，该差异仍存在。

[`gemm_diagnostic.json`](gemm_diagnostic.json) 用捕获的同一份第一层投影输入和权重，
脱离调度器与 KV，重放 `F.linear`：一次计算 1024 行与两次各计算 512 行相比，
1,048,576 个输出元素中有 1,207 个不一致，最大差异 0.0078125。原投影输出与重放的
1024 行调用精确一致，因此这一处分叉可独立复现为 BF16 GEMM 行形状相关的舍入差异，
而不是已证实的 KV 地址错误。不能据此声称后续所有差异都已逐层定位。

统一使用 512 行计算块的**孤立线性层控制**在 batch=8/4 间精确一致，提供后续修正
方向；尚未接入完整模型或通过全步复验，不能宣布已修好。FP32 只作 GEMM 诊断参考，
不进入 BF16 性能统计。两个诊断都不会放行原失败 gate。

下一步需要在独立策略/目录中统一 Prefill 线性层计算形状，检查所有层、完整输出预算
与新到达场景；通过后才能重新启动 18-process 性能对照。原失败证据保持不变。

后续完整策略已在独立的 [`h2_scheduling_tile512/`](../h2_scheduling_tile512/README.md)
通过全部 1,664 个向量对照与 18/18 性能进程、审计。本轮原 harness、两项诊断及配置的七个文件已冻结在
[`source_snapshot/manifest.json`](source_snapshot/manifest.json)，均匹配原始 SHA256。
当前可演进代码不再代表本轮源码版本；本轮回放/核对以该快照为准。原 gate、报告、
诊断、参考张量及完整性审计不修改，旧失败也不会被新策略追认为通过。

计划正式范围为 3 场景 × 2 策略 × 3 独立进程，每进程 3 次 trace 回放；目前未启动。
证据路径约定如下，`raw/summary/audit` 性能证据尚未产生：

- `preflight/<policy>.json`、`preflight.json`：跨策略数值、状态和哈希门禁；
- `raw/<case>/<policy>/run{1,2,3}.json`：接入时间、逐 token 时间戳、每步调度、完整请求预算与输出 hash；
- `summary.json`、`audit.json`：独立进程统计量中位数及原始记录重算审计；
- 日志与 BF16 参考张量分别放在忽略的 `runs/logs/`、`runs/validation/`，不提交大文件。

这里只比较接入策略，不代表已完成分页容量/碎片、抢占压力、Chunked Prefill、TP、
HTTP 稳态负载或整个 H 阶段。全部旧结果与实施大纲保持不变。

回归：116 项中 114 项通过，2 项 CUDA/Triton 专项在隐藏 GPU 的 CPU 环境跳过。
GPU 探针及两项定位均已退出，GPU 已释放。CPU 调度仿真只证明状态和指标定义，
不作为性能结果。失败证据完整性与旧结果保全检查另见 `integrity_audit.json`。

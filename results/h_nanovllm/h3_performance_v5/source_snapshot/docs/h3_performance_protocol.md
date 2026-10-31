# H3：同模型、同环境、全 BF16 的性能对照

2026-10-24 冻结 v1 范围。配置：[h3_performance_v1.json](../config/serving/h3_performance_v1.json)。
本协议定义新实验；原 H3 正确性结果与 `performance_execution_allowed=false` 保持原样。

## 问题、路径与边界

在同一张 RTX 5060 Laptop GPU、原 serving 环境、同一个 Dense artifact 下，对照：

| 路径 | 模型与 KV | 执行方式 |
|---|---|---|
| native_kv | 原生 GPT、连续 KV | eager |
| paged_eager | 现有 RouteScale bridge、上游分页 attention 与调度 | eager |
| paged_graph | 同一 bridge、同一分页池 | Decode CUDA Graph；Prefill 与 LM head 仍在图外 |

三侧全部使用 BF16 权重，关闭 autocast 与 Prefix，TP=1，保留 learned absolute
positions、LayerNorm/GELU 与 tied embedding。artifact、tokenizer、数据、请求、配置、
源码、依赖和 GPU 信息均记录哈希或版本。保持已有环境、上游与权重不变。
native/paged 对照同时改变 attention 实现、布局和控制路径，不能把全部差异归因于分页。
paged eager/Graph 是本轮更直接的执行图对照。

## 输入、采样与测量含义

固定 validation token 窗口，同时冻结 prompt 和逐步回放的后续 token。所有路径每一步
输入相同，不根据输出或速度挑选窗口。每个请求都达到固定输出预算，忽略 EOS；最后一个
输出不再送入模型，因此必须满足 `prompt + output - 1 <= 512`。

分别测两种范围，使用相同路径、相同输入与独立预热：

1. `model_execution`：CUDA Event 包围模型调用和全词表 logits；输入准备、请求组批、
   argmax、CPU 输出与资源回收在区间外。它是 CUDA Event 的执行区间，包含可能的提交空隙，
   不是 Profiler 的纯 kernel busy-time 总和。
2. `replay_e2e`：从请求入队／native cohort 建立开始，经组批、H2D、模型、全词表
   argmax 与 `.tolist()` CPU token 交接，回放固定 token，直至请求完成和资源回收。
   TTFT/ITL/TPOT 均按 CPU 可见 token 时间计算，E2E 在 GPU 同步后结束。

第二项是共同的**固定 token 回放 harness**，不冒充上游公共 `generate()`、自然生成、
tokenizer/text 输出、HTTP 服务或在线 SLO。argmax 结果只用于诊断，后续输入使用冻结的
validation token；不会用相同随机 seed 假定不同实现的采样轨迹相同。吞吐计数为实际
完成的回放输出 token 数。两种范围单独汇总，不能交叉相除求“框架加速比”。

## 先验正确性门槛

在新范围中新进程生成原生全 BF16 full-recompute 参考，保存每请求、每步的 50257 维
logits。三条目标路径也分别用新进程验证全部步骤：finite、`atol=0.125, rtol=0.02`，
每向量 relative RMSE ≤0.005；eager/Graph 全部向量还须 bitwise exact。
门槛沿用旧正确性合同，不在看到结果后放宽。

每条路径共 864 个向量，三侧共 2592 个；原生参考另有 864 个。NaN poison KV 后运行，
验证缓存长度、零 Prefix 命中、一次完整 Prefill、跨页增长与结束后的资源释放。
核对三侧 BF16 参数哈希与权重共享。批量 Prefill 上限提升到 2048，覆盖 8×255；
原 1024-token gate 不作为新范围通过的替代证据。新 gate 通过后方可启动 pilot。

## 矩阵、预热与统计

| case | B | prompt | output | 目的 |
|---|---:|---:|---:|---|
| latency_short | 1 | 32 | 32 | 单请求提交开销 |
| batch_short | 8 | 128 | 64 | 批量 Decode |
| page_growth | 8 | 255 | 32 | 实际跨页增长 |
| long_prefill | 1 | 384 | 64 | 较长 Prefill |

分页池固定 16×256 tokens，即 64 MiB。native 预先分配 B×512-token 连续缓存，B=1
为 8 MiB，B=8 为 64 MiB。实际分配与预算分别报告，不声称所有形状物理分配相同。
加载、上游 warmup、Graph capture、固定池分配计入初始化并单独报告，排除出稳态时间。

pilot 覆盖 latency_short/page_growth×三路径，共六个新进程；每种测量范围预热三轮、
测五轮，检查 wall-time CV≤15%、输出预算、页增长、释放与显存余量≥512 MiB。
若不通过，保留失败，不启动正式矩阵；改变范围或策略须新配置与新目录。

正式为 4 case×3 路径×3 独立进程，共 36 个。每进程两种测量范围各预热三轮、测五轮。
每个 run 轮转三条路径的先后顺序，单 GPU 串行，避免固定顺序偏置。
保留逐轮／逐 token 计时，报告各进程统计值、三次中位数及离散程度；样本分位数只作
本地回放诊断，不解释为在线用户尾延迟。吞吐统一为 `B×output / E2E seconds`，
TPOT 统一为 `(last_token_time-first_token_time)/(output-1)`。

独立汇总重算全部原始指标与比例，核对 source/config/gate/workload/tensor 哈希、
请求预算、逐轮确定性与无资源泄漏；保存负结果。原 G3 与 Qwen 历史数字不进入排名。

## 文件与执行

- 新结果：[h3_performance_v1/](../results/h_nanovllm/h3_performance_v1/)。
- 请求：`runs/workloads/h3_performance_v1/requests.json`；完整 logits：`runs/validation/h3_performance_v1/`。
- 日志／编译缓存：`runs/logs/h_nanovllm/h3_performance_v1/`、`runs/cache/h3_performance_v1/`。
- 失败与原始记录不覆盖；每次新尝试使用新配置与结果／workload／validation 目录。

```bash
.venv/bin/python scripts/run_h3_performance.py --dry-run
.venv/bin/python scripts/run_h3_performance.py
runs/envs/nano-vllm/bin/python scripts/audit_h3_performance.py --check-only
```

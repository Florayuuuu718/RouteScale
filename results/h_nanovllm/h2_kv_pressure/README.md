# H2：KV 压力下的真实请求接入与吞吐

状态：[run.json](run.json) 为 pass，18/18 正式进程及[审计](audit.json)通过。
六个独立 GPU 预检进程的[新门禁](preflight.json)包含 4,160 个逐请求全词表向量，
全部 exact；最终审计重读两侧完整保存张量再次精确对齐。
正式计时共 54 条 trace、864 个请求、74,880 个输出 tokens。

本轮延续[分配容量实验](../h2_kv_capacity/README.md)，但执行的是真实模型、真实
sampler、完整 16 请求及其全部输出预算。完整变量和计时口径见
[协议](../../../docs/h2_kv_pressure_protocol.md)。

## 预先固定的对照

| 场景 | 每侧实际 KV tensor | 输出/请求 | 固定预留 / 分页可资助 cohort |
|---|---:|---:|---:|
| tight_short | 448 MiB | 64 | 4 / 8 |
| tight_growth | 448 MiB | 132 | 4 / 8 |
| ample_short | 896 MiB | 64 | 8 / 8 |

两侧固定相同模型、16 个 128-token prompt、BF16、TP=1、共同 resident 上限 8，
相同 paged attention、Decode Graph、Prefill tile512 与 FCFS fixed-cohort 调度。
只改预留策略和由 KV 预算推导的接入数，不把连续补位收益混进来。
这里限制的是实际 KV tensor 配额，未耗尽整卡显存，也不是整卡最大容量搜索。

固定预留每请求拥有 1024-token 窗口的四页；分页按需追加，但先按
`ceil((prompt+output-1)/256)` 为峰值增长计入信用。最后一个采样 token 尚未进入
下一次 forward，故 132-token 输出最后对应 259-token KV。增长场景从 8 个已用
页实际增长到 16 个，而不是只观察 prompt 刚放下时的容量。

## 实测结果

下表来自 [summary.json](summary.json)：先重算每个进程三条 trace 的统计量，再取
三个独立进程的中位数。TTFT/E2E 从整个突发到达开始，包含未接入请求的等待。

| 场景 | 策略 | 输出 tokens/s | TTFT p95 ms | TPOT p50 ms | E2E p95 ms | 峰值已用页 |
|---|---|---:|---:|---:|---:|---:|
| tight_short | static_window | 540.08 | 1558.69 | 6.29 | 1991.80 | 16 |
| tight_short | paged | 1053.23 | 583.83 | 6.35 | 1009.80 | 8 |
| tight_growth | static_window | 593.41 | 2834.59 | 6.16 | 3659.21 | 16 |
| tight_growth | paged | 1119.57 | 1041.72 | 6.42 | 1958.21 | 16 |
| ample_short | static_window | 1080.12 | 557.66 | 6.31 | 961.01 | 32 |
| ample_short | paged | 1005.61 | 676.23 | 6.47 | 1086.38 | 8 |

紧预算下，真实 cohort 从 4 增到 8，短请求/增长请求吞吐分别为 **1.950× / 1.887×**，
TTFT p95 分别下降 62.5% / 63.2%。这不是 attention kernel 变快，而是少预留空间
转成更大的有效 batch。单请求 TPOT p50 却增加约 0.9% / 4.2%，不能把总吞吐增加
解释成每个请求吐 token 都更快。

优势针对的是“整窗口预留”这个明确基线。如果连续分配也利用已知最大输出长度来
缩小预留，接入差距和收益可能改变；本轮不能证明分页胜过所有按需连续分配器。

更大 batch 也使峰值 allocated 从 1635.17 增至 1653.18 MiB，增加 18.01 MiB；
KV tensor 同为 448 MiB，不是减少了物理池。增长场景确实在 KV=257 时从 8 页
追加到 16 页，后续所有输出仍完成，没有抢占、OOM 或引用残留。

### 充裕预算的负结果

ample_short 两侧均为 batch=8，每条 trace 相同两次 Prefill/126 次 Decode，每对同 seed 的
采样输出哈希都一致。分页本轮吞吐却只有 **0.931×（−6.9%）**，TTFT p95 为
1.213×，E2E p95 为 1.130×。三次独立进程吞吐：固定预留
1080.12 / 1037.12 / 1092.60，分页 1005.61 / 1011.65 / 987.69 tokens/s。
不能删掉这个负对照，也不能未经定位便断言它只是噪声。

原始时间边界显示部分差异在 `model_start_ms → token_visible_ms` 内，该区间包含
输入准备、模型、采样、copy 和 postprocess，**不是纯 GPU kernel 时间**。例如
[paged run1](raw/ample_short/paged/run1.json) 第二条 trace 的两次 Prefill 合计
290.34 ms，而[固定预留 run1](raw/ample_short/static_window/run1.json) 三条 trace
的对应合计为 146.67～153.20 ms。计时前后编译计数均为 21，没有发生新编译；
但这些 wall-time 记录还不能区分页地址访问、设备状态或主机调度的贡献。
后续应单独诊断这项退化，不把“已用逻辑页更少”直接等同于更快。

后续独立[配对与页地址诊断](../h2_kv_diagnostic/README.md)已完成：原 0.931× 退化
在平衡配对中未稳定复现，Decode stream 布局中位差约 1%。该诊断没有覆盖本表，
也没有唯一确认历史根因；这里的原始负结果继续保留。

两侧该场景 peak allocated 同为 2101.18 MiB，物理 KV tensor 同为 896 MiB；
分页虽只拥有 8 页，未分配给请求的页仍在预先申请的物理池里。

## 测量与审计边界

每场景/策略三个独立进程，每进程一条预热 trace、三条正式 trace；第二个独立 run
反转策略顺序。TTFT/E2E 从所有请求同时到达开始，包含尚未接入时的排队等待；
TPOT/ITL 使用实际 token 可见时间。初始化、capture、warmup、reset、输出哈希排除，
内部 trace 记账包含在 wall time 中。编译前后计数必须不变，不允许数值 probe 混入
性能结果。完整 BF16 probe 张量放在忽略的 `runs/validation/h2_kv_pressure/`。

最终审计重读两侧完整张量、重算 raw 指标，核对每一步 KV 页数、峰值信用、释放
状态，以及旧 18+36+18 份正式 raw。充裕预算负对照的模型调用结构必须相同。
采样输出哈希只作诊断；不同批大小可能改变 RNG 消耗，不能据此做质量判断。
432 对请求中有 216 对采样哈希一致：ample_short 为 144/144，两个紧预算场景各
36/144；这与 4,160 个完整 teacher-forced 向量精确一致并不矛盾。

全部 18 个进程的编译计数在正式计时前后均为 21，512 MiB 安全 headroom 约束通过。
新增 16 项单测通过；CPU 全量回归 152 项，150 通过、2 项 CUDA/Triton 按 CPU 模式
跳过。只读复验入口为 `scripts/summarize_h2_kv_pressure.py --check-only`。

两侧仍使用相同分页 attention 内核：这不是“原生连续 attention vs 分页 attention”
微基准，也不包含未知输出长度、抢占/换出、连续补位、HTTP/SLO 或最大并发搜索。
教学大纲与历史原始结果保持不动。

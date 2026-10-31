# H2：固定 cohort 与连续接入对照

本轮继续既有 H2，研究请求接入策略，不训练或更换模型，不修改实施大纲、上游 checkout
或已经通过的 Graph/Prefix 实验代码。失败版本源码先精确冻结再演进。原配置为
[`h2_scheduling.json`](../config/serving/h2_scheduling.json)，新策略配置为
[`h2_scheduling_tile512.json`](../config/serving/h2_scheduling_tile512.json)，当前状态见
[`新 run.json`](../results/h_nanovllm/h2_scheduling_tile512/run.json)。

历史版本：CPU 准备与 GPU 主预检已完成，混合场景未通过既定数值门槛，正式进程
为 0/18。首个分叉已定位至 Prefill `o_proj` 的 BF16 GEMM 行形状；仅孤立线性层的
512 行分块控制通过，尚无完整模型修复或性能排名。数据、负结果与下一步见
[`本轮记录`](../results/h_nanovllm/h2_scheduling/README.md)。

后续 `prefill_linear_tile512_v1` 已接入完整模型并通过两个独立进程的复验，全部
1,664 个全词表向量精确一致；新目录的 18/18 正式进程、结果审计和独立验证全部通过，见
[`新策略记录`](../results/h_nanovllm/h2_scheduling_tile512/README.md)。旧 gate 仍为 fail。
混合请求吞吐约提高 32%，TTFT p95 降低约 42% / 56%，但 ITL p99 从约 8.5 ms 升至
71～75 ms；等长负对照没有机制收益。本轮为本机有限 trace 对照，不是在线 SLO。

## 唯一改变的条件

- `fixed_cohort`：最多接入 8 个已到达请求；这一组全部结束后，才接下一组。
- `continuous`：每个模型迭代前检查空位和已到达请求，有空位就补入，最多 8 个在途请求。

两侧均沿用上游紧凑 Decode：短请求结束后立即返回结果并释放资源，不继续计算 padding，
不人为延迟短请求返回。所谓 fixed 是 **cohort 接入不补位**，不是静态满 batch 张量。
上游 `Scheduler.max_num_seqs` 只约束一次调度数量，不能代替在途请求总量限制；
因此两侧共用一个显式 admission 控制层。请求排序均为 FCFS，不预知输出长度来排序。

模型、权重 revision、tokenizer、BF16 和隔离环境沿用 H1。两侧均为
`static_tables_paged_prefill_v1`、Graph 开启、Prefix 关闭、TP=1、32×256 KV blocks
（896 MiB）、1024 模型窗口、4096 batched tokens、8 sequences。所有 prompt 均为
128 tokens，输出最长 64，最多 8 个活跃请求仅需 8 个 blocks；这轮没有缓存压力、
抢占、分页碎片、Chunked Prefill 或 HTTP 服务方面的结论。

## 三个固定到达场景

复用已有 fixture `b8_p128_o64_none` 的 cold/warm 共 16 个唯一 prompt，不重新采样
输入。新 workload hash 同时绑定全部 prompt、请求顺序、计划到达时间和输出预算。

| 场景 | 到达时间 | 每组 8 请求的输出长度 | 用途 |
|---|---|---|---|
| equal_burst | 16 请求同时到达 | 全部 32 tokens | 无空槽可补时的结构性负对照 |
| mixed_burst | 16 请求同时到达 | 前 4 个 8、后 4 个 64 tokens，两组重复 | 检查短请求结束后能否利用空槽 |
| mixed_staggered | 首 8 个在 0 ms，后 8 个在 160 ms | 同上 | 检查运行途中新请求的排队与 Prefill 干扰 |

`ignore_eos=true`、temperature=0.6，使用上游真实 sampler；不要求跨策略采样输出
一致，因为批组成和调用次数会改变 RNG 消耗顺序。固定长度只用于隔离调度，不是
依据已知长度进行优先级调度，也不是文本质量评测。

CPU 状态测试中，mixed_burst 的 fixed 为 2 次 Prefill + 126 次 Decode，continuous
为 3 次 Prefill + 77 次 Decode；后者平均 Decode batch 更大，但额外 Prefill 会
打断旧请求。这只是调度步数检查，不是 GPU 性能证据，也不能据此预先宣布提速。

## 数值与状态门禁

旧 Graph / Prefix gate 不直接放行本轮。先在两个独立 GPU 进程中检查全部场景，
每请求每 token 对齐相同的反向 prompt teacher-forced 续写。fixed 的 BF16 logits
张量存入忽略的 `runs/validation/h2_scheduling/<results_dir>/`；continuous 逐向量与之
比较，共 1,664 个完整词表向量，涵盖每个请求的全部 8/32/64 步。

继续使用原事前阈值：atol=0.05、rtol=0.02、relative RMSE≤0.005，同时报告精确相等
步数，不放宽误差预算。比较包含不同批组成、旧请求 KV 保留、新请求 Prefill、
请求结束及资源回收；先将整个 KV tensor 填为 NaN，再检查所有 logits 有限。
gate 还绑定模型/环境、旧修正策略、新源码/配置、case hash、实际 KV tensor、
数值报告与参考张量的 SHA256。失败立即停止，不产生正式倍率。

探针复制 logits 到 CPU 会改变真实时间，因此 **数值探针使用每次模型迭代 10 ms
的逻辑时钟**回放到达事件，而不是让调试开销决定接入顺序。它不是性能样本。
实际计时完全移除探针、forced token 和 logits CPU copy，按真实主机时钟回放计划
到达，继续检查活跃请求上限、FCFS、接入时机、输出预算、零命中和全部引用释放。

## 性能计时与尾延迟

三场景 × 两策略 × 三独立进程 = 18 个正式进程，单卡串行，按 run 轮换策略顺序。
每进程先完整回放一次作为 warmup，再测三次固定 trace；每次从新 BlockManager
开始，物理 KV tensor 不变。初始化、Graph capture、编译、warmup、cache reset、
输出 hash、写盘均不计入时延；计时窗口前后编译计数必须不变，禁止静默 fallback。

TTFT/E2E 从 **计划到达时间**计算，不从真正入引擎的时间计算。请求在 GPU 步执行
期间到达，驱动会在迭代边界发现它；这一发现/等待时间仍包含在时延里，不把晚入队
伪装成晚到达。

- admission delay：计划到达到真正入引擎；
- scheduler wait：入引擎到首次进入模型；
- TTFT/E2E：计划到达到首/末 token 主机可见；
- TPOT：`(末 token 时间 − 首 token 时间)/(输出数−1)`；
- ITL：同一请求相邻 token 的实际间隔，另报 p50/p95/p99/max；
- 吞吐：总完成请求/输出 tokens 除以全部 trace wall time，包含到达等待；
- mean Decode batch：Decode 迭代 batch size 的平均值，不是 GPU 利用率；
- peak allocated：真实模型、固定 KV tensor、Graph 与临时 tensor 的峰值，不是 reserved。

上游优先安排 Prefill，不能同时声称补位必然改善吞吐、TTFT 与 ITL。新请求的 Prefill
可能降低排队时间，却增加旧请求的 token 间隔。这里只是本机有限到达 trace 的离线
回放，不是 HTTP、持续稳态 QPS、饱和点或在线 SLO；不把这些尾延迟冒充线上结果。
每进程内合并三个 trace 的样本，最终取三个独立进程统计量的中位数，不混成全局 p99。

## 复现与证据位置

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests
runs/envs/nano-vllm/bin/python scripts/run_h2_scheduling.py --config config/serving/h2_scheduling_tile512.json --preflight-only
# 只有新的 preflight.json 通过后，才续跑正式矩阵。
runs/envs/nano-vllm/bin/python scripts/run_h2_scheduling.py --config config/serving/h2_scheduling_tile512.json --resume
runs/envs/nano-vllm/bin/python scripts/summarize_h2_scheduling.py --config config/serving/h2_scheduling_tile512.json
runs/envs/nano-vllm/bin/python scripts/audit_h2_scheduling_tile512.py
```

runner 与既有 H2 入口共用单实例锁，已有 raw 不覆盖；修复或重新实验需另开
`results_dir` 重新验收，不用 resume 绕过失败。
日志写入 `runs/logs/h_nanovllm/h2_scheduling/`；小体积证据进入
[`results/h_nanovllm/h2_scheduling/`](../results/h_nanovllm/h2_scheduling/README.md)。

原配置/目录保留失败版本；当前显式传入 `h2_scheduling_tile512.json`，新日志及参考
张量使用新目录名。原源码快照和 manifest 见历史目录 `source_snapshot/`；当前源码
的版本变化不会把旧失败追认为成功。已经通过的 Graph/Prefix 实验代码保持原哈希。

### 完整模型的线性层 shape 策略

新策略在 28 层、112 个 QKV/O/gate-up/down 投影中，把 Prefill GEMM 的输入行数统一
为 512；1024 行拆为两个 512 行调用，512 行仍为一次，不增加 padding 或改变请求。
数学操作、权重和 BF16 不变，实际浮点计算路径与原生 GEMM 不同，因此两侧同时应用、
新 gate/来源信息单独绑定，并单独测性能。不能将这一实现的结果当成无 adapter 的
上游性能，也不声称更接近 Transformers 参考或模型质量更好。

Decode 原样运行 Graph；adapter 在 Graph capture 后安装，只处理这轮 512/1024 行
Prefill，超出形状或精度范围直接报错。关闭引擎前恢复全部 bound methods 并释放
闭包，避免保留模型引用。新门禁在保留原容差字段之外，进一步要求全步 **exact**。
原生 1024 行线性层不能默认与两个 512 行调用 bitwise 等价；只修改 continuous 一侧
会同时改变计算路径，失去受控对照。

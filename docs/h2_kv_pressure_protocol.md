# H2：KV 预算约束下的真实模型接入与吞吐

配置：[h2_kv_pressure.json](../config/serving/h2_kv_pressure.json)。
前置证据：[KV 容量与碎片检查](../results/h_nanovllm/h2_kv_capacity/README.md)。
本轮从逻辑分配检查走到真实模型请求，但不使用 CPU 模拟耗时作为 GPU 性能。

## 变量与范围

固定 Qwen3-0.6B revision、BF16、TP=1、1024-token 窗口、256-token 页、相同
attention 路径、Decode Graph 与 Prefill tile512 控制。共同 resident 上限和引擎
`max_num_seqs` 均为 8。16 个固定 128-token prompt 在 trace 开始时同时到达。

两侧都是 FCFS fixed-cohort：等整批结束再接下一批，不混入连续补位调度收益。
只改变预留/接入方式：

- `static_window` 每请求实际拥有完整窗口的四页，未来页私有保管，当前页表只暴露
  已用部分；复用上一轮通过 GPU 地址与完整模型检查的 adapter。
- `paged` 使用上游按需分配，但接入前用已知 `max_tokens` 为峰值增长计入逻辑信用。
  信用占用并不提前写入真实 block table；Decode 时仍由上游真实追加页。

峰值 KV 为 `prompt + output - 1`，因为最后一个采样 token 尚未经过下一次 forward。
按需策略不能只用 prompt 页数来接入请求。若 FCFS 队首无法获资，直接报错，不跳过
它，也不等到 Decode 无页时才把抢占或死锁算入这轮无抢占基准。

| 场景 | 两侧真实 KV 池 | 输出/请求 | 固定预留 / 分页可资助批大小 | 用途 |
|---|---:|---:|---:|---|
| tight_short | 16 页、448 MiB | 64 | 4 / 8 | 验证少预留能否转为真实批处理收益 |
| tight_growth | 16 页、448 MiB | 132 | 4 / 8 | 分页在 257-token KV 处增长至满 16 页 |
| ample_short | 32 页、896 MiB | 64 | 8 / 8 | 引擎上限约束下，无接入机会差异的负对照 |

每个 pair 的物理池完全相同；不同池只在不同场景中出现，不拿跨预算差异冒充策略
收益。这是人为固定配额的 KV 池压力，不是耗尽整张 GPU 的显存；设备仍保留充足
安全空闲，因此不据此宣称整卡的最大服务容量。两侧仍都使用 paged attention，
不是原生连续 attention kernel 对比。
这不是最大并发搜索：只验收上述 4/8-request cohort、16-request 本地突发。

## 数值门槛与正式矩阵

新结果目录与新 gate，不继承上一轮容量 pass 作为吞吐许可：

1. 三场景各两个独立 GPU 进程，共六个。每个实际执行全部 16 请求、全部输出步，
   cyclic-reversed prompt teacher forcing。KV 先填 NaN，逐步检查 BF16 全词表
   logits 有限。不同批大小会改变形状，因此要求 4160 个逐请求向量全部 exact。
2. 两侧保存完整 logits 到忽略的 `runs/validation/h2_kv_pressure/`，JSON 只保存
   路径和 SHA256。gate 绑定模型、环境、旧/新代码、请求与每场景有效引擎参数。
3. gate 通过后运行 `3 场景 × 2 策略 × 3 独立进程 = 18` 个正式进程；第二个
   独立 run 反转两策略顺序。每进程一次完整 workload warmup，三条正式 trace。
4. 正式过程使用真实 sampler，固定 ignore_eos/max_tokens，按 trace 重置 RNG。
   不抓取 logits、不 teacher force；编译计数在计时前后必须不变，禁止静默 fallback。
5. 计时从全体请求到达/开始 enqueue 前算起，TTFT/E2E 包含尚未接入的等待；包含
   排队、调度、输入准备、模型、采样、token copy、postprocess 与 trace 记账。
   排除初始化/Graph capture/编译、workload warmup、cache reset 和输出哈希。
6. 按实际 token 可见时间记录 TTFT、TPOT、ITL、E2E，同时记录逐步活跃请求、真实
   used/free 页、峰值信用页与当前 KV 长度。审计按 token 位置独立重放，而非相信
   单个 `peak_blocks` 字段。每条 trace 结束后队列/引用全释放。

汇总器重算每个原始 trace 的指标，再取三个独立进程统计量的中位数；不把所有进程
的请求混成一个伪大样本。最终独立审计还会重新读取两侧完整保存张量比较全部向量，
核对旧 18+36+18 raw、充裕池两侧模型调用结构与教学大纲无修改。

## 入口与输出目录

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_h2_kv_pressure
runs/envs/nano-vllm/bin/python scripts/run_h2_kv_pressure.py --preflight-only
runs/envs/nano-vllm/bin/python scripts/run_h2_kv_pressure.py --resume
runs/envs/nano-vllm/bin/python scripts/summarize_h2_kv_pressure.py --check-only
```

| 内容 | 位置 |
|---|---|
| journal、六份 probe、gate、18 份 raw、summary/audit | `results/h_nanovllm/h2_kv_pressure/` |
| 完整 logits | `runs/validation/h2_kv_pressure/h2_kv_pressure/` |
| 独立子进程日志 | `runs/logs/h_nanovllm/h2_kv_pressure/` |

runner 使用公共 GPU 锁；任何失败保留原证据，不覆盖失败记录或降门槛。
未覆盖未知输出长度、提前 EOS 的真实流量、连续补位、抢占/换出、compaction、HTTP
SLO、多卡与 RouteScale 模型接入。采样哈希只作诊断：批形状变化可能改变 RNG 消耗，
不能把不同策略的采样文本差异当成模型质量差异或绕过完整 logits 门槛。

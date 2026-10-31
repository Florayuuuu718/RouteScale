# H2 调度对照：完整 Prefill 线性层形状一致性策略

日期：2026-10-23。**完整模型数值验收和 18/18 正式性能进程全部通过**，原始指标
重算与独立验证也通过。数值门禁共 1,664 个全词表向量逐步精确一致；正式测量共
54 次 trace、864 个请求、29,952 个输出 tokens，不包含 teacher-forced 探针。
记录见 [`run.json`](run.json)、[`preflight.json`](preflight.json)、
[`summary.json`](summary.json)、[`audit.json`](audit.json)、[`verification.json`](verification.json)。
模型仍为固定 revision 的 Qwen3-0.6B；没有训练、换模型或改精度。

## 性能结果及作用

单张 RTX 5060 Laptop GPU（8151 MiB），两侧使用相同的完整线性层策略。下表为三个
独立进程统计量的中位数；每进程有三次 trace，不是全局线上尾延迟或纯 GPU kernel 时间。

| 到达场景 | 输出吞吐 fixed → continuous（tokens/s） | 吞吐倍率 | TTFT p95（ms） | ITL p99（ms） |
|---|---:|---:|---:|---:|
| 等长突发 | 960.57 → 931.81 | 0.970× | 353.81 → 391.76 | 8.11 → 8.18 |
| 长短混合突发 | 587.52 → 775.88 | 1.321× | 583.10 → 337.20 | 8.46 → 75.01 |
| 长短混合延迟到达 | 586.08 → 773.42 | 1.320× | 461.76 → 204.62 | 8.47 → 70.80 |

混合请求中，补位把平均 Decode batch 从 4.44 提高到 7.27 / 6.86，吞吐提高约 32%；
TTFT p95 降低 42.2% / 55.7%。三次 trace 合计的 Decode 步数从 378 降到 231 / 245
（独立进程中位数），不是靠换模型、增加在途槽位、加大 KV pool 或计算 padding 得到。
两侧峰值 allocated 均为约 2101.18 MiB，物理 KV tensor 均为 896 MiB。

代价是旧请求的 token 输出被新 Prefill 打断：混合场景 ITL p99 升至约 71～75 ms，
ITL max 从 12.44 → 82.21 ms、9.67 → 98.78 ms。`mixed_burst/continuous/run1` 的第一
次 trace 中，两次补入 Prefill 约耗时 73 / 70 ms，原长请求 `r04` 对应的相邻 token
间隔约 80 / 76 ms。上游优先处理 Prefill、此间不执行旧请求 Decode，解释了排队和
吞吐改善但 ITL 尾部退化的现象。这也说明 TPOT 的小幅变化不能代替逐 token 尾延迟。

等长负对照没有补位机会：两侧模型步顺序完全相同，平均 Decode batch 都是 8，144
组配对采样输出全部一致；本轮约 3% 的吞吐波动不能解释为补位收益，也不足以据此
断言稳定损失。只有三次独立进程，不能把每个小幅差异都作确定的因果归因。

因此在本项目中已分别得到三条证据：Graph 主要减少本机 Decode 执行开销，Prefix
复用减少可复用输入的 Prefill，而 continuous admission 利用空出的在途槽位；最后
一项还会影响正在输出的旧请求。三者都不能仅凭总吞吐决定在线用户体验。

跨策略采样 RNG 消耗顺序不同：432 组配对请求中，等长 144/144 输出 hash 一致，
两种混合场景各 36/144 一致。**这不是数值误差率或质量得分**；数值门禁使用同一续写，
全部 1,664 向量精确一致。采样输出只用于资源/预算验收与诊断。

本轮在完整模型的 28 层、112 个 QKV/O/gate-up/down 线性投影上统一
`prefill_linear_tile512_v1`：128-token 请求的 4/8 批 Prefill 总行数为 512/1024，
均由相同的 **512 行 BF16 GEMM**组成，不填充、不重排 token、不改权重、bias 或 KV。
1024 行调用拆为两个真实 512 行调用；512 行调用保持一次。Decode Graph 路径不变。
该策略是显式的可复现 adapter，不是上游原生性能；两种调度策略都应用，不能只修
continuous 一侧。安装发生在上游模型 warmup / Graph capture 后，workload warmup 前。

前一轮[失败记录](../h2_scheduling/README.md)与 gate 保持不变。旧 harness、两项诊断
及配置的七个文件已精确冻结于
[`source_snapshot/manifest.json`](../h2_scheduling/source_snapshot/manifest.json)，各源码
SHA256 与原 gate/诊断匹配；当前 harness 在此基础上演进，新结果只写入本目录。
已通过的历史 Graph/Prefix 实验源码不受这次扩展影响。

## 冻结的验收与性能范围

仍使用三个原场景、16 个固定 prompt、8 个在途槽位、Graph 开启、Prefix 关闭、
32×256 物理 KV blocks（896 MiB）。请求、计划到达、输出预算和计时边界均不变。
fixed cohort 不计算虚假 padding，已完成请求立即返回；continuous 只改变补位时机。

新门禁在两个独立 GPU 进程中，对同样的完整 teacher-forced 续写逐请求比较全词表，
共 1,664 个向量（512 + 576 + 576），本轮额外要求 **全部逐步精确一致**，而非仅满足
原 atol=0.05、rtol=0.02、relative RMSE≤0.005。NaN-poison 有限性、实际 KV 容量、
在途上限、每个输出预算、FCFS 和最终资源释放也必须通过。

数值探针使用逻辑时钟，正式计时使用真实主机到达时间和原生 sampler；不混用。
通过后才运行 3 场景 × 2 策略 × 3 独立进程，共 18 个正式进程，每进程 3 次 trace。
初始化、编译、Graph capture、warmup、cache reset、输出 hash 与写盘排除在正式计时外。
TTFT/E2E 从计划到达算起；另报 admission delay、scheduler wait、真实逐 token ITL
及平均 Decode batch。完整口径见[调度协议](../../../docs/h2_scheduling_protocol.md)。

```bash
runs/envs/nano-vllm/bin/python scripts/run_h2_scheduling.py --config config/serving/h2_scheduling_tile512.json --preflight-only
runs/envs/nano-vllm/bin/python scripts/run_h2_scheduling.py --config config/serving/h2_scheduling_tile512.json --resume
runs/envs/nano-vllm/bin/python scripts/summarize_h2_scheduling.py --config config/serving/h2_scheduling_tile512.json
runs/envs/nano-vllm/bin/python scripts/audit_h2_scheduling_tile512.py
```

日志写入 `runs/logs/h_nanovllm/h2_scheduling_tile512/`，BF16 参考张量写入忽略的
`runs/validation/h2_scheduling/h2_scheduling_tile512/`；小体积 JSON/Markdown 留在本目录。
代码或配置改变会使 gate/续跑失效，不能覆盖旧 raw。该 adapter 限定本轮 512/1024 行
Prefill 和 TP=1，不代表任意批大小、在线 HTTP/SLO、容量压力或多卡已经通过。

回归：123 项中 121 项通过，2 项 CUDA/Triton 专项在隐藏 GPU 的 CPU 环境跳过。
独立验证确认所有进程均安装 112 个线性 adapter、无 padding、全步 logits exact、
raw/summary 重算一致、模型/环境/参考/源码哈希匹配、等长负对照步结构相同；历史
18 + 36 份正式 raw 和原失败 gate/源码快照仍匹配。实施大纲未改，GPU 已释放。

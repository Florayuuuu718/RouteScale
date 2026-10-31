# nano-vLLM：H1 复现、H2 消融与 H3 本项目接入

最新进展（2026-10-24）：[H3 同模型受控性能 v5](h3_performance_v5/README.md)已完成
2592 个 BF16 数学参考比较、864 个 Graph exact 向量、6/6 pilot、36/36 正式进程和
独立审计。三条路径统一关闭 BF16 GEMM 中间累积降精度许可；v1–v4 数值失败保留。
观察到 Graph 提高回放吞吐而 TTFT 不改善，但计时波动较大，追加三组配对诊断仍未
唯一定位历史原因。稳定性能幅度需要复验；不是公共生成 API、在线 SLO 或整个 H 阶段完成。

较早进展：[H3 首轮本项目 Dense 接入](h3_routescale/README.md)已通过
三变体正确性与独立复核：792 个原生 BF16 全词表向量通过，264 个 eager/Graph 向量
bitwise exact，运行时 52 个权重张量键一致，公共请求保护与资源回收通过。本轮运行的
是本项目训练权重，保留 learned absolute positions / LayerNorm / GELU；不改旧 Qwen
路径。该首轮不包含正式计时；后续受控结果见上方 v5，不代表整个 H 阶段完成。协议见
[`h3_routescale_protocol.md`](../../docs/h3_routescale_protocol.md)。以下保留 H1/H2 历史。

日期：2026-10-23。**H1 全部通过**：安装与环境检查、模型快照校验、11 项状态测试、
FP16/BF16 attention 检查和两个离线请求均完成。
完整执行记录见 [`h1_run.json`](h1_run.json)。H2 的 batch=8 CUDA Graph 独立对照已完成，
18/18 正式进程通过，本机吞吐为 eager 的 6.21～8.51×，详见
[`h2_graph_b8/README.md`](h2_graph_b8/README.md)。Prefix 和 batch=1 Graph 数值 gate
未通过的原始记录均保留。后续两项差异已在显式新策略下修正并通过 2,048 组完整
64 步 logits 验证，见 [`h2_numerics/README.md`](h2_numerics/README.md)。新策略的 36-process
首轮性能矩阵与审计已全部通过：本机 Graph 吞吐为 eager 的 5.90～7.47×；有效 Prefix
命中使 warm TTFT 从 137.64 ms 降至 71.57 ms，但整段吞吐基本不变。详见
[`h2_corrected_pilot/README.md`](h2_corrected_pilot/README.md)。这只覆盖原四 case 的
Graph / Prefix 两项消融，不代表连续批处理、碎片/容量、多卡 TP 或整个 H 阶段完成。

随后已建立 fixed-cohort / continuous-admission 接入对照。三个场景的 GPU 新预检中，
等长场景 512 个全词表向量精确一致，混合场景最大 relative RMSE 3.974%，未通过
0.5% 门槛；正式进程 0/18。第一处分叉定位为 Prefill 线性层随行形状变化的 BF16
舍入，原失败保留于 [`h2_scheduling/README.md`](h2_scheduling/README.md)。
后续完整模型 tile512 策略已通过 1,664 个全步向量精确对照及 18/18 性能进程、独立
审计。混合请求吞吐约提高 32%，TTFT p95 降低约 42% / 56%，但 ITL p99 从约 8.5 ms
升至 71～75 ms；等长负对照无机制收益。详见
[`h2_scheduling_tile512/README.md`](h2_scheduling_tile512/README.md)。这是共同 adapter
下的本机对照，不是原生上游性能或在线 SLO。

随后[KV 容量与碎片检查](h2_kv_capacity/README.md)完成：18 个 CPU 容量点、13 个
真实 GPU tensor 地址/隔离检查，跨过 256-token 页边界的 1,056 个全词表向量全部
exact。896 MiB 固定池中，128-token 当前 KV 的逻辑容纳量由固定预留的 8 个变为
分页的 32 个；257-token 点则暴露明显页尾浪费。连续空洞与分页复用分别验证，未把
逻辑容量当作吞吐或实测服务并发。

随后[真实模型 KV 配额实验](h2_kv_pressure/README.md)完成 4,160 个 exact 向量及
18/18 正式进程、审计，共 864 请求/74,880 输出 tokens。448 MiB 紧预算下，真实
接入批由 4 增至 8，短请求/跨页增长吞吐为 1.950× / 1.887×；但 896 MiB 充裕预算
负对照只有 0.931×，不能声称分页普遍更快。两侧仍用相同 paged attention、已知
输出上限下预资助增长，无抢占/换出或整卡最大容量结论；该轮不覆盖多卡与 H3 接入。

随后[负对照诊断](h2_kv_diagnostic/README.md)完成三个独立进程、48 条平衡配对完整
请求 trace、27 条页地址 stream 探针、3072 个 exact 向量。配对吞吐比中位数
1.017×，原 0.931× 退化未稳定复现；Decode 布局差约 1%，部分 Prefill 差异仍在。
这不是修复或覆盖旧结果，历史具体根因未唯一确认。没有改分配器或电源设置；随后
H3 首轮正确性接入已独立完成，详见页首链接。

H1/H2 固定对象：nano-vLLM `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（MIT），
Qwen/Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`（Apache-2.0）。
源码保持独立 clean checkout，权重、环境与编译缓存均放在忽略的 `runs/` 下。

## 已完成的状态验收

[`h1_state_tests.json`](h1_state_tests.json) 保存 11 项直接针对上游实现的测试：

- block 分配/回收与空闲、使用集合的资源守恒；
- 活跃 prefix 的引用计数共享，以及已释放 prefix 的复活；
- 淘汰后旧 hash 的移除、prefix hash 链和 partial tail 不共享；
- block 耗尽与 Decode 跨页时的 admission 检查；
- Decode 资源不足时的 scheduler 抢占；
- chunked Prefill 只在最后一个 chunk 完成后追加生成 token；
- EOS 与最大生成长度结束后的请求/资源回收。

CPU 测试直接加载 pinned upstream 文件，只跳过会提前导入 GPU engine 的包初始化。
它们验证状态机，不证明 attention kernel 的数值正确性或 GPU 性能。
这 11 项测试已在最终锁定环境重新执行并全部通过。

## H1 GPU 验收

[`h1_attention.json`](h1_attention.json) 在 RTX 5060 Laptop GPU 上验证真实内核：

| dtype | Prefill 最大绝对误差 | Paged Prefix Prefill | Paged Decode | Triton KV 写入 |
|---|---:|---:|---:|---|
| BF16 | 0.007244 | 0.003615 | 0.002051 | 精确一致 |
| FP16 | 0.000957 | 0.000481 | 0.000481 | 精确一致 |

参考为 FP32 SDPA，验收 atol/rtol 均为 0.03。这是 attention 层的数值检查，不是整个
Qwen 模型相对 Transformers 的 logits 等价证明。

[`h1_smoke.json`](h1_smoke.json) 保存两个原生 API 请求的输出。第一个由 EOS 结束，
第二个达到 64-token 上限；结束后 running/waiting 队列和 used blocks 均为 0。
GPU utilization budget=0.5 时实际分配 58 个 KV blocks，测得峰值 allocated 约
2821.9 MiB。初始化 21.27 秒、首次生成 23.33 秒包含 JIT 等冷启动开销，仅作诊断，
不作为正式吞吐结果或生成质量评估。

## H2 已完成的 CPU 准备

- [`h2_workload_manifest.json`](h2_workload_manifest.json)：24 组固定 token workload，
  cold/warm 请求各不相同；完整数组留在忽略的 `runs/workloads/`。
- [`h2_prefix_probe.json`](h2_prefix_probe.json)：48 组真实上游元数据 probe 全部通过。
  128-token high prefix 命中为 0；512-token high prefix 在 warm 波每请求复用 256
  tokens，即输入命中率 50%；关闭 prefix 时均为 0。
- [`h2_execution_plan.json`](h2_execution_plan.json)：保留的历史 CPU 计划，曾被原数值
  gate 拦截；不是当前进度。新策略 36-process 状态见 `h2_corrected_pilot/run.json`。
- 汇总器通过单元测试，会拒绝 CPU evidence、不同模型或 pool、缺失 run、
  输出预算变化、非法延迟/命中数据。GPU 计时入口与共同物理 pool adapter 已实现。

## H2 数值门槛与首轮拆分

原 36-process 计划在正式计时前被严格 Prefix gate 拦截，见
[`h2_pilot/preflight.json`](h2_pilot/preflight.json)。BF16 Prefix warm logits 最大差异
0.3125、relative RMSE 最高 2.02%，不满足事前阈值；没有放宽阈值或把失败改为 pass。

[`prefix_numerical_diagnostic.json`](h2_pilot/prefix_numerical_diagnostic.json) 中，28 层
复用 KV 均与 cold 值精确一致，A/A logits 精确一致，NaN-poisoned cache 检查通过。
第一层 Q/K/V 相同但 attention 输出开始有差异；仅将 attention 换成 FP32 SDPA math
参考后，最终 logits 差异为 0。该证据支持 BF16 attention 路径的数值差异，不是质量
评估，也不把 FP32 参考速度混入原生 BF16 数据。后续修正仍保留 BF16，只统一原本不同
的 attention 路径；新策略的 Prefix 正式性能对照已在独立目录完成，详见上方链接。

四 case Graph-only 的 [完整数值诊断](h2_graph_pilot/all_case_numerical_diagnostic.json)
发现 batch=1 Decode 最大差异 0.25390625，并有一步 top-1 分叉，因此原 24-process
范围也在计时前停止。该预检时尚未定位；后续页表宽度修正见新策略记录，不能沿用 Prefix 的根因解释。

batch=8 的三个 case 全部 24 个张量精确一致，独立配置为
[`h2_graph_b8.json`](../../config/serving/h2_graph_b8.json)，在新目录重做 preflight 后
运行 3 × 2 × 3 = 18 个正式进程。两者均关闭 prefix，固定相同 32-block/896 MiB KV
tensor。该 scope 的 pass 不覆盖 batch=1 / Prefix Cache，也不是完整 H2 通过。

本轮 18/18 已完成，[审计](h2_graph_b8/audit.json)确认 raw/summary 重算一致，720 对
采样输出哈希一致，源码和权重未变化。Graph 峰值 allocated 增加约 8.14 MiB；512-token
TTFT 基本不变，收益集中在 Decode。详细数值和不能外推的范围见该轮 README。

详见 [`docs/h2_ablation_protocol.md`](../../docs/h2_ablation_protocol.md)。

## 依赖处理

首次尝试 FlashAttention 2.8.3.post2 源码安装时，本机 nvcc 为 13.0，训练环境
PyTorch 为 2.12.1+cu129，构建因 CUDA major 版本不匹配停止；官方 release 也未提供
torch2.12 对应 wheel。H1 因此采用独立 Python 3.11 / PyTorch 2.8.0+cu129 环境，配套
官方 torch2.8 wheel，避免修改训练环境。

FlashAttention wheel 已通过官方发布 SHA256
`92aabf18ecfcf5d620ac7fe97fafe88f83e714b7a790e4bf219dff4a404cfb38`。
[`h1_wheel_inspection.json`](h1_wheel_inspection.json) 进一步确认二进制内包含 sm_120
代码。GPU kernel 与模型运行的结论分别写入 `h1_attention.json`、`h1_smoke.json`；
H1 初始化/首次生成耗时不进入正式排名。

## 复现与学习入口

完整环境、配置和命令见 [`docs/h_stage_protocol.md`](../../docs/h_stage_protocol.md)。
源码调用链、资源生命周期与 G 阶段对照见
[`docs/nanovllm_source_walkthrough.md`](../../docs/nanovllm_source_walkthrough.md)。

三个执行入口均在项目根目录使用独立环境运行：

```bash
runs/envs/nano-vllm/bin/python -m serving.verify_nanovllm_state
runs/envs/nano-vllm/bin/python -m serving.verify_nanovllm_attention
runs/envs/nano-vllm/bin/python -m serving.reproduce_nanovllm
```

H1 通过后再进入 H2 的 CUDA Graph / Prefix Cache 正式单变量实验。

已启动的下载可由 `python -m serving.run_h1 --wait-for-downloads` 自动衔接安装与验收。
它校验所有大文件 SHA256，逐步执行并将日志写入 `runs/logs/h_nanovllm/`。

# H2 新策略性能对照

日期：2026-10-23。**36/36 正式进程与后置审计全部通过**，见
[`run.json`](run.json)、[`summary.json`](summary.json) 和 [`audit.json`](audit.json)。
共测量 2,250 个请求、144,000 个输出 tokens；每项机制各 750 组配对请求的采样输出
哈希全部一致。数值验收的 teacher-forced 请求不计入这些性能样本。

这轮固定使用 `static_tables_paged_prefill_v1`：所有变体都使用完整模型窗口宽度的
block table 和 paged Prefill，BF16、模型和请求不变。前置门禁为
[`validation_v1/gate.json`](../h2_numerics/validation_v1/gate.json)，2,048 个完整词表
logits 张量对照精确一致。这是受控新策略的消融，不是未加 adapter 的 upstream
性能，也不能继承 [`历史 batch=8 结果`](../h2_graph_b8/README.md) 的倍率。

## 正式结果及如何理解

以下均为 **warm 波、三个独立进程统计量的中位数**；cold/warm 全部原始值、
TTFT/TPOT/E2E p50/p95/p99 和峰值显存保存在 summary/raw，不合并成在线尾延迟。

| case | 输出吞吐：eager → Graph（tokens/s） | Graph 吞吐倍率 | TTFT p50：eager → Prefix（ms） | Prefix 输入命中率 |
|---|---:|---:|---:|---:|
| b1 / p128 / none | 19.15 → 143.04 | 7.47× | 63.58 → 57.76 | 0% |
| b8 / p128 / high | 150.73 → 1086.80 | 7.21× | 58.32 → 65.89 | 0% |
| b8 / p512 / none | 139.91 → 824.92 | 5.90× | 137.59 → 137.83 | 0% |
| b8 / p512 / high | 141.91 → 839.09 | 5.91× | 137.64 → 71.57 | 50% |

Graph 在四 case 的 cold/warm 共八个对照中，输出吞吐为 eager 的 **5.90～7.47×**。
收益集中在 Decode；不能解释为 Prefill 计算也等比例加速。512-token TTFT 基本不变，
batch=1 warm TTFT 甚至从 63.58 ms 上升至 72.52 ms。所有 Graph 对照的峰值 allocated
增加 8.144 MiB，KV pool 不变。初始化中位数约为 eager 5.79 s / Graph 10.52 s，其中
Graph capture 中位数 4.79 s；另有单独记录且不计入正式测量的 workload warmup。

有效 Prefix 命中每请求复用 256 tokens，warm TTFT 降低 **48.0%（1.92×）**，
但 TPOT 为 57.38 → 57.25 ms，整段吞吐为 141.91 → 141.10 tokens/s（0.994×），
没有对应的整体吞吐收益。固定生成 64 tokens 时，Decode 仍占据主要时长。
该 warm 波峰值 allocated 少 56.04 MiB，但 **896 MiB 的物理 KV tensor 完全不变**，
不能据此宣称 KV pool 缩小或已测得更大的最大并发。

负结果同样保留：短共享前缀无完整页可复用，warm 吞吐为基线的 0.899×；无共享前缀
组也有上下波动。零命中时的变化不能解释为缓存复用收益；仅三个进程重复，不能将
每个小幅变化都归因于某一项开销。实际学习到的区分是：Graph 主要改善本机 Decode
执行开销，Prefix 复用主要改善可复用输入的 Prefill/首 token 时延，两者不保证同时
改善吞吐、TTFT 和显存。

本轮仅完成四 case 下 Graph / Prefix 两项受控消融；连续批处理、分页碎片/容量、
多卡 TP 与 RouteScale Dense 模型接入仍是后续工作，不宣称整个 H 阶段完成。

## 固定实验范围

| case | batch | prompt / output tokens | Prefix warm 每请求预期命中 |
|---|---:|---|---:|
| `b1_p128_o64_none` | 1 | 128 / 64 | 0 |
| `b8_p128_o64_high` | 8 | 128 / 64 | 0（公共前缀不足完整页） |
| `b8_p512_o64_none` | 8 | 512 / 64 | 0 |
| `b8_p512_o64_high` | 8 | 512 / 64 | 256（输入命中率 50%） |

每个 case 比较 eager/no-prefix、Graph/no-prefix、eager/prefix；各三个独立进程，
共 36 个正式进程。模型为固定 revision 的 Qwen3-0.6B；TP=1，单卡串行。
每个变体均实际创建 32 个 256-token KV blocks，共 896 MiB KV tensor；Graph 不通过
缩小 KV pool 获得显存空间。每进程一个完整 cold/warm warmup pair 不计时，再测五个
pair。每对开始前重置缓存元数据，cold/warm 之间保留；输出 64 tokens、ignore_eos。

硬件为单张 RTX 5060 Laptop GPU（8151 MiB）。报告包含 enqueue、调度、输入准备、
模型、采样、token 拷贝与 postprocess 的主机可见
离线批时延，不是线上 SLO。初始化、编译、Graph capture、warmup、cache reset 和
输出 hash 不计入正式时延；初始化和 capture 耗时单独记录。全部分位数、吞吐与
显存的定义见 [`H2 协议`](../../../docs/h2_ablation_protocol.md)。

## 结果与复现

- `raw/<case>/<variant>/run{1,2,3}.json`：每请求时延、命中、采样输出 hash、编译计数及释放状态。
- `summary.json`：三个独立进程统计量的中位数，cold/warm 分开；不是合并样本的全局 p99。
- `audit.json`：重算 raw 和 summary，校验固定 pool、精确命中、模型/源码/环境以及 paired 输出 hash。
- 日志：忽略的 `runs/logs/h_nanovllm/h2_corrected_pilot/`。

```bash
runs/envs/nano-vllm/bin/python scripts/run_h2_corrected_pilot.py
# 中断后，先检查 run.json 和日志；仅复用同一来源且通过检查的 raw。
runs/envs/nano-vllm/bin/python scripts/run_h2_corrected_pilot.py --resume
runs/envs/nano-vllm/bin/python scripts/audit_h2_corrected_pilot.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

新入口与历史入口共用单实例锁；失败即停止，不覆盖已有 raw。门禁与性能入口各有
独立源码哈希；代码或模型变化必须在新目录重新验收。所有变体均开启编译上限失败
检查，禁止静默 fallback；计时窗口内新增编译也视为失败。原生路径的两份 fail gate
和已经完成的 18-process 历史结果保持不变。

回归检查：104 项测试在可见 GPU 环境全部通过；隐藏 GPU 的 CPU 复跑中 102 项通过，
2 项 CUDA/Triton 专项按环境跳过。历史 18 份 raw 与原源码、新策略数值 gate 与三个
数值报告的 SHA256 均匹配原记录；两份历史 native gate 仍为 fail。实验与审计结束后 GPU 已释放。

# H3 首轮：本项目 Dense 权重真正接入 nano-vLLM

日期：2026-10-24（北京时间）。**接入正确性及独立复核通过；未开展正式性能计时。**
后续独立的 [v5 受控性能矩阵](../h3_performance_v5/README.md)已执行 36/36 进程并通过
审计，计时稳定性限制单列；本目录仍保留首轮正确性证据，不覆盖旧 gate。

这次运行的是 RouteScale 自己训练的 TinyStories Dense checkpoint，不是 Qwen 示例。
没有重新训练或修改原 FP32 artifact，也没有修改上游 checkout、两个环境的依赖或教学大纲。

配置见 [`h3_routescale.json`](../../../config/serving/h3_routescale.json)，详细合同见
[`h3_routescale_protocol.md`](../../../docs/h3_routescale_protocol.md)。

## 已测结果

| 检查 | 覆盖与结果 |
|---|---|
| 架构/权重映射 FP32 | 三变体共 42 个全词表向量通过；最大绝对误差 2.862e-6 |
| 分页执行 vs 原生 BF16 全量重算 | 792 个全步全词表向量通过；最大绝对误差 0.0625，最大 relative RMSE 0.3431%，门槛 0.5% |
| eager vs CUDA Graph | 264 个向量 bitwise exact |
| Prefix cold/warm | 仅开启 Prefix 的 warm 波每请求命中 256 tokens，关闭时为零 |
| 跨页与窗口边界 | 两个请求从各 255 增至 262 cached tokens，总已用页从 2 增至 4；窗口 case 达到 512 cached tokens |
| 离线 GPT-2 tokenizer | 12 个编码/解码样例与源 tiktoken 一致，每个变体均检查 |
| 公共文本生成 | 两请求各生成 8 tokens，结束后无活跃请求/页引用；只作链路 smoke |
| 独立参考复核 | 新进程重新计算原生 BF16 参考，全部 792 个向量再次通过 |
| 运行时权重 | 52 个 state_dict 张量键与原 checkpoint 的 BF16 转换逐项相等；embedding/head 仍共享 |
| 公共请求保护 | 6 个非法请求在入队前拒绝；512-token prompt + 1 输出真实生成通过 |
| 独立 eager 引擎生命周期 | 关闭后 runner/model/KV 弱引用均回收；本进程 allocated 回到 111,821,824 bytes 基线 |

最后一行的基线包含独立原生参考模型，并非 CUDA context 或 allocator reserved cache
全部为零。进程退出后本轮 GPU 占用已释放。

三种独立变体为 `eager_no_prefix`、`graph_no_prefix`、`eager_prefix`；没有联合
Graph+Prefix 变体。KV tensor 固定 16 页、每页 256 tokens，共 **64 MiB**，不是
搜索整卡最大容量。各变体结束后 running/waiting 队列、used blocks 和引用均回收。
另核对四轮历史 H2 的 90 个正式 raw JSON 哈希，均未变化。

## 本项目如何使用上游能力

[`PackedRouteScaleGPT`](../../../serving/h3_model.py) 保留原 learned absolute position
embedding、LayerNorm、GELU、QKV/投影与 tied LM head，只将 packed attention 对接上游
KV 页表。没有把 GPT 改成 Qwen/RoPE。显式
[`bridge`](../../../serving/backends/routescale_nanovllm.py) 复用上游 Scheduler、
BlockManager、Sampler、Prefill/Decode 组批和 Decode CUDA Graph；绕过硬编码 Qwen 的
构造阶段，不改变旧 Qwen 路径。

模型为 8 层/8 头/宽 512，词表 50257。计入 learned position embedding 后，独立参数
为 51,168,256；历史 50.91M 为不计位置 embedding 的口径，不是本轮换了模型。

位置预算为 `prompt_tokens + max_tokens - 1 <= 512`。最后采样 token 不再入模，所以
最终序列可以有 513 tokens，但 cached positions 仍最多 512。不满足预算时拒绝请求，
**不会**静默滑窗或采用 G2/G3 的 cache rebuild。

## 证据入口

- [`preparation.json`](preparation.json)：离线 tokenizer、请求与 builder 哈希。
- [`run.json`](run.json)：preparation、三变体、compare 全部完成。
- [`eager_no_prefix.json`](eager_no_prefix.json)、[`graph_no_prefix.json`](graph_no_prefix.json)、
  [`eager_prefix.json`](eager_prefix.json)：逐 case 数值、页表、命中和文本 smoke。
- [`gate.json`](gate.json)：三进程共同 provenance、792 / 264 数值范围和历史哈希。
- [`verification.json`](verification.json)：独立原生参考、权重/API/生命周期验收。

完整 logits 与独立参考只放忽略的 `runs/validation/h3_routescale/`；tokenizer/请求在
`runs/workloads/h3_routescale/`，日志与缓存也位于 `runs/`，不将这些大文件加入结果目录。

只读重核命令（项目根目录）：

```bash
runs/envs/nano-vllm/bin/python -m serving.verify_h3_routescale --check-only
runs/envs/nano-vllm/bin/python scripts/audit_h3_routescale.py --check-only
```

CPU 回归共 171 项：169 项通过，2 项 CUDA/Triton 测试因该次显式禁用 CUDA 而跳过；
本轮新增 9 项全部通过。上述真实 GPU 接入验收独立执行，不依赖这两个跳过项。

## 不做的外推

原生参考和桥接目标均为 serving 环境的全 BF16 权重。G3 的 FP32 权重加 BF16 autocast
以及 Qwen 的历史结果，均不是本轮直接性能对照。`performance_execution_allowed=false`，
没有 TTFT/TPS 排名，没有“分页或 Graph 一定加速”的结论。

尚未覆盖 MoE、TP>1、超窗口重建、到达流/抢占压力或在线 SLO；文本 smoke 不证明
生成质量提高。下一步是先冻结**同模型、同环境、同精度**的原生/桥接计时合同，
通过新范围的正确性门槛后，再做正式性能比较。

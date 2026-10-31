# H3：RouteScale Dense 接入 nano-vLLM 的正确性协议

日期：2026-10-24（北京时间）。本轮完成**第一范围的架构/权重/数值接入**，不代表
H3 性能实验、整个 H 阶段或 miniSGLang 已完成。结果见
[`h3_routescale/README.md`](../results/h_nanovllm/h3_routescale/README.md)。

## 固定对象与精度

- 使用 G1 导出的本项目 TinyStories Dense checkpoint，不再用 Qwen 代替本项目模型。
  原文件为 `runs/checkpoints/g_serving/dense_baseline/model.pt`，SHA256 为
  `1c3b6835dd3c394a617cccc3c398a277e6fb99e6a1026680c14dfd1b1d946d2a`。
  不重新训练、不重新导出、不修改原 FP32 artifact。
- 原结构：8 层、8 头 MHA、宽度 512、head_dim=64、词表 50257、learned absolute
  position window=512、原生 LayerNorm 和 GELU。全部独立参数为 51,168,256；
  历史的 50.91M 口径不计 262,144 个位置 embedding 参数。
- 上游固定为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，不改其 clean checkout。
  TP=1，不要求四卡，也不建立多卡进程组。
- 数值参考与目标都在独立 serving 环境中使用**全 BF16 权重**：PyTorch 2.8.0+cu129、
  FlashAttention 2.8.3.post2。G3 是训练环境的 FP32 权重加 BF16 autocast，环境与精度
  合同均不同，因此不能直接拿 G3 或 Qwen 的历史 tokens/s 排名。
- 配置见 [`h3_routescale.json`](../config/serving/h3_routescale.json)。模型、配置、
  workload、tokenizer、上游与 adapter 源码、软件包列表均写入 provenance。

## 哪些部分保留，哪些部分适配

[`PackedRouteScaleGPT`](../serving/h3_model.py) 直接持有 strict-load 后的原生 GPT，
没有将它转换成 Qwen。它将不同请求压成一维 token/position 数组，保留每个请求从零
开始的绝对位置；forward 中只替换 attention 的执行与 KV 寻址。

| 原生 GPT 部分 | 接入处理 |
|---|---|
| token embedding + learned position embedding | 保留原 `wte`、`wpe` 与逐请求 position IDs，不引入 RoPE |
| LayerNorm、QKV linear、输出投影、残差 | 直接调用原模块，不转换为 RMSNorm、不重排或转置权重 |
| MLP | 直接调用原模块，保留 GELU，不改为 SwiGLU |
| causal attention | 对 packed Q/K/V 调用上游 Attention，以物理页表读写 KV |
| 输出头 | Prefill 取各请求最后一个 query；保留原词表和 embedding/head 权重共享 |
| 请求执行 | 复用上游 Sequence、Scheduler、BlockManager、Sampler、prepare/run 与 Decode Graph |

上游 `ModelRunner.__init__` 硬编码 Qwen，`LLMEngine.__init__` 又依赖 HF 自动配置。
[`routescale_nanovllm.py`](../serving/backends/routescale_nanovllm.py) 因此提供显式 opt-in
构造器，只绕过这两段初始化，不 monkey-patch 旧 Qwen 入口。固定物理 KV tensor 为
`[2, 8, 16, 256, 8, 64]` BF16，共 64 MiB，每页 4 MiB。

本轮显式复用 H2 的固定页表宽度与统一 paged Prefill 策略；**没有套用 Qwen 的
tile512 全模型线性层策略**。这属于项目桥接路径，不能称为上游原生 RouteScale 支持。

## 离线 tokenizer 与输入合同

[`h3_prepare.py`](../serving/h3_prepare.py) 在未改动的训练环境中读取已有 GPT-2
tiktoken 字节/rank 表，导出 BPE vocab、merges、regex 与 special IDs；serving 只用
已有的 `tokenizers` 构建对应分词器。不新增软件包、不下载新模型或 tokenizer。

12 个编码/解码样例覆盖 G1 文本、中文、emoji、重音字符、空白、控制字符、换行和
缩写，逐一与原 tiktoken 对照。词表仍为 50257，EOS 为 50256；这只是覆盖样例的
一致性证据，不声称穷举所有 Unicode 输入。

公共 `add_request` 在入队前拒绝空输入、越界/非法 token IDs、非正整数输出预算、
非法 temperature 和超窗口预算。遵循上游 `temperature > 1e-10` 合同，不支持
temperature=0。special-token 文本默认拒绝，与源 tiktoken 默认行为一致。

窗口判断是 `prompt_tokens + max_tokens - 1 <= 512`：最后一次采样出来的 token
没有进入下一次 forward/KV。因此 512-token prompt + 1 输出合法，+ 2 输出拒绝；
最终 prompt+output 总长度可以为 513，**已进入模型的缓存位置仍不超过 512**。
本轮不实现 G2/G3 的滑窗截断与 cache rebuild，也不静默更改位置编码。

## 预先固定的验收范围

三种变体各用一个独立 GPU 进程：`eager_no_prefix`、`graph_no_prefix`、`eager_prefix`。
这是功能矩阵，不是三次性能重复；没有 Graph+Prefix 同时开启的范围。

| case | 各请求 prompt tokens | 每请求输出 tokens | 目的 |
|---|---|---:|---|
| short | 32 | 16 | 单请求基础路径 |
| batch | 128 / 128 / 128 / 128 | 32 | 等长 packed 批次 |
| ragged | 17 / 63 / 129 / 257 | 8 | 不等长分段与绝对位置 |
| page_cross | 255 / 255 | 8 | Decode 跨过 256-token 页边界 |
| window_edge | 505 | 8 | 最后一次 forward 恰好达到 512 cached tokens |
| prefix_reuse | 257 / 257 | 16 | cold/warm 两波；warm 时每请求命中 256 tokens |

数值验证使用固定的 teacher-forced 后续 token，避免采样分叉改变比较输入；每一步
保存全部 50257 维 logits，不以 top-1 相同或文本“看起来合理”代替数值验收。
文本 API 另做两个真实采样 smoke 请求，各生成 8 tokens，不用于质量或性能评价。

验收分为三层：

1. FP32 架构映射：用分段 SDPA 参考替代 attention，每个 prompt 对比原生 GPT
   full recomputation。要求 finite 且 `atol=5e-5, rtol=1e-5`，每变体 14 个向量。
2. BF16 分页/Prefix：真实上游 attention/KV 执行逐步对照同环境、同 BF16 权重的
   原生 GPT full recomputation。要求 finite、`atol=0.125, rtol=0.02` 且每向量
   relative RMSE 不超过 0.005。每变体 264 个向量，合计 792 个。
3. Graph：保存的 eager 与 Graph 全步张量须 bitwise exact，共 264 个向量。

执行前用 NaN 填充 KV，检查实际命中、页表增长、结束后队列与 block 引用回收。
Prefix warm 必须实测每请求 256 cached tokens；关闭 Prefix 的两种变体必须为零。
数值门槛不得在看到结果后放宽。

## 独立复核与结果保存

主 runner 完成 preparation、三进程与 compare 后，另以
[`audit_h3_routescale.py`](../scripts/audit_h3_routescale.py) 启动新进程，独立重算
每请求、每一步的原生 BF16 参考，再核对全部 792 个目标向量。另新建一个 eager
bridge，比较 52 个 state_dict 张量键、权重共享、6 个非法公共请求、512 cached-token
合法边界，以及关闭后的 runner/model/KV tensor 弱引用回收。52 个键包含共享 embedding/head
的两个别名，不表示存在 52 组互不共享参数。

轻量证据放在 `results/h_nanovllm/h3_routescale/`；大文件分工为：

- `runs/workloads/h3_routescale/`：离线 tokenizer 与固定请求数组；
- `runs/validation/h3_routescale/h3_routescale/`：保存的 logits，子目录
  `independent_native/` 保存独立重算的参考；
- `runs/logs/h_nanovllm/h3_routescale/`：各子进程日志；
- `runs/cache/h3_routescale/`：本轮独立编译缓存。

已有结果不覆盖。默认配置已经完成，直接重复 runner 会拒绝执行。新尝试须另存配置，
同时选择新的 `results_dir`（末级目录名也须唯一）与 `workload_dir`，再执行：

```bash
.venv/bin/python scripts/run_h3_routescale.py --config <新的配置路径>
runs/envs/nano-vllm/bin/python scripts/audit_h3_routescale.py --config <新的配置路径>
```

对本轮现存结果只读重核，不重新训练或覆盖证据：

```bash
runs/envs/nano-vllm/bin/python -m serving.verify_h3_routescale --check-only
runs/envs/nano-vllm/bin/python scripts/audit_h3_routescale.py --check-only
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests
```

`--check-only` 重查保存张量与哈希，不重新运行公共 API/生命周期测试；这些初次新
进程的实测结果保留于 `verification.json`。核查仍要求原机 serving 环境及本地大文件。

## 尚未成立的结论

本轮 `performance_execution_allowed=false`。尚未冻结同环境、同 dtype 的原生连续 KV
与桥接分页 KV 正式计时合同，不能据此发布 TTFT、TPS 或显存容量排名，也不能直接
复用 G3 的历史计时作对照。下一范围应先定义共同请求、相同采样、预热/重复与计时
边界，再通过该范围的新正确性 gate 后测量。

本轮还不覆盖 MoE、TP>1、Graph+Prefix 联合变体、超窗口重建、任意到达流、抢占压力、
在线服务/SLO 或模型生成质量提升。基础模型规模和训练规划不在本协议内变更。

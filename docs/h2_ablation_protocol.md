# H2 CUDA Graph / Prefix Cache 实验协议

请求配置位于
[`config/serving/h2_nanovllm.json`](../config/serving/h2_nanovllm.json)。模型、tokenizer
revision、上游 commit 和环境沿用已通过验收的 H1。
原三变体单卡 GPU 首轮配置位于
[`config/serving/h2_pilot.json`](../config/serving/h2_pilot.json)，执行状态见
[`h2_pilot/run.json`](../results/h_nanovllm/h2_pilot/run.json)。该轮被 Prefix Cache 数值
gate 拦截，未启动正式矩阵。四 case 的 Graph-only 预检又发现 batch=1 Decode 分叉。
历史上先完成数值已通过的 batch=8 三 case。两项差异在显式新策略中修正后，另开
原四 case 的三变体性能对照，不替换失败点、不放宽阈值。

| 配置 / 结果目录 | 范围 | 状态 |
|---|---|---|
| `h2_pilot.json` / `h2_pilot/` | 三变体、原计划 36 进程 | Prefix warm 数值 gate 失败，未启动正式计时 |
| `h2_graph_pilot.json` / `h2_graph_pilot/` | Graph-only、原计划 24 进程 | batch=1 Decode gate 失败，未启动正式计时 |
| [`h2_graph_b8.json`](../config/serving/h2_graph_b8.json) / [`h2_graph_b8/`](../results/h_nanovllm/h2_graph_b8/) | batch=8 三 case、两变体、三个独立 run | 18/18 正式进程与结果审计通过 |
| [`h2_corrected_pilot.json`](../config/serving/h2_corrected_pilot.json) / [`h2_corrected_pilot/`](../results/h_nanovllm/h2_corrected_pilot/) | 新策略、原四 case、三变体、三个独立 run | 36/36 正式进程与结果审计通过 |
| [`h2_scheduling.json`](../config/serving/h2_scheduling.json) / [`h2_scheduling/`](../results/h_nanovllm/h2_scheduling/) | fixed cohort / continuous admission，三个到达场景 | 混合场景新数值 gate 失败，正式进程 0/18；Prefill 线性层形状差异已定位 |
| [`h2_scheduling_tile512.json`](../config/serving/h2_scheduling_tile512.json) / [`h2_scheduling_tile512/`](../results/h_nanovllm/h2_scheduling_tile512/) | 完整 BF16 线性分块、相同三个到达场景 | 1,664 全词表向量 exact，18/18 正式进程、结果审计与独立验证通过 |
| [`h2_kv_capacity.json`](../config/serving/h2_kv_capacity.json) / [`h2_kv_capacity/`](../results/h_nanovllm/h2_kv_capacity/) | 固定预留 / 精确连续参考 / 分页的容量、空洞与增长 | 18 个 CPU 容量点、13 个真实 GPU 地址检查、1,056 个跨页向量 exact；不是性能矩阵 |
| [`h2_kv_pressure.json`](../config/serving/h2_kv_pressure.json) / [`h2_kv_pressure/`](../results/h_nanovllm/h2_kv_pressure/) | 相同 KV 配额、真实模型、峰值增长资助、三个场景 | 4,160 个向量 exact，18/18 正式进程及审计通过；充裕预算负结果保留 |
| [`h2_kv_diagnostic.json`](../config/serving/h2_kv_diagnostic.json) / [`h2_kv_diagnostic/`](../results/h_nanovllm/h2_kv_diagnostic/) | 充裕预算负结果的配对/页地址诊断 | 48 条完整请求 trace、27 条 stream 探针、3072 个向量 exact；未稳定复现旧退化，不替换正式结果 |

历史 batch=8 子集本身不代表 batch=1 或 Prefix Cache 通过；新策略单独完成这两项。
Graph/Prefix 首轮本身不覆盖连续批处理、分页碎片/容量与多卡 TP；后续连续接入另有
独立 scope。后续分页分配容量/碎片与受控 KV 配额下的真实请求性能已单独验证，
但抢占/换出、未知输出长度及多卡等未覆盖，不能宣称完整 H2 或 H 阶段完成。
历史合格子集的结果与适用边界见
[`h2_graph_b8/README.md`](../results/h_nanovllm/h2_graph_b8/README.md)。

后续：两个数值问题已在显式新策略 `static_tables_paged_prefill_v1` 下完成修正复验，
8 个 case、64 步 cold/warm、2,048 个全词表张量对照精确一致，见
[`h2_numerical_fixes.md`](h2_numerical_fixes.md) 与
[`新策略验收`](../results/h_nanovllm/h2_numerics/validation_v1/gate.json)。
下文保留原失败的历史诊断；它们不改成 pass，新策略也不继承历史吞吐数字。
新策略四 case 的 36 个正式进程及审计已通过：2,250 个请求、144,000 个输出 tokens，
Graph / Prefix 各 750 组采样输出哈希一致。Graph 吞吐 5.90～7.47×；有效 Prefix
warm TTFT 137.64 → 71.57 ms，但整体吞吐约 0.994×。完整数据、负结果和限制见
[`新策略性能记录`](../results/h_nanovllm/h2_corrected_pilot/README.md)。

后续连续接入实验的变量、请求到达和逐 token 计时口径另见
[`h2_scheduling_protocol.md`](h2_scheduling_protocol.md)。该轮不继承本轮 gate：不同
接入批大小会改变 Prefill GEMM 行形状，现已发现新的跨策略数值边界；完整修正通过前
不运行其性能矩阵，也不影响已经完成的 Graph/Prefix 证据。
后续完整 tile512 策略已在新目录完成该复验与性能矩阵：混合请求吞吐提高约 32%，
TTFT p95 改善但 ITL 尾部退化。旧失败仍保留，当前代码与原失败版本的源码快照
分别绑定不同 gate，不混用原生路径和 adapter 路径的性能数据。

再后续的[KV 容量协议](h2_kv_capacity_protocol.md)区分固定窗口预留浪费、分页内部
页尾浪费、连续分配外部空洞。两种 GPU 预留策略仍走相同 paged attention，完整
模型跨页向量全部 exact。该 scope 只证明分配/地址/数值，不签发吞吐测量 gate，
也不把 32 个逻辑分配称为已完成 32 并发服务。

后续另开[KV 压力性能协议](h2_kv_pressure_protocol.md)，以实际 16/32 页 tensor
配额比较真实模型请求，未耗尽整张 GPU 显存。新 gate 的 4,160 个向量 exact，
18 个正式进程与审计通过。紧预算接入批增大有收益，充裕预算负对照吞吐 0.931×，
尚不能将该退化归因于页地址或主机状态。它不继承容量实验的 pass 作为性能许可。

后续[负对照诊断协议](h2_kv_diagnostic_protocol.md)用同进程平衡顺序和受控页号复查：
三个进程的配对吞吐比中位数 1.017×，原 −6.9% 未稳定复现。Decode stream 中位差
约 1%，Prefill 仍有局部差异；尚不能唯一归因历史系统状态，不删除或重算旧负结果。
teacher-forced stream 探针不是正式服务 tokens/s，也不签发新的性能排名。

## 请求集合与 CPU 证据

完整 fixture 提供 batch=1/8/32、prompt=128/512、output=64/256，以及 none/high
prefix，共 24 个 case。固定英文语料用本地 Qwen tokenizer 编码，按固定 seed 构造
精确长度的 token 数组，不加载模型。每个 case 有 cold/warm 两波请求，通过唯一尾部
标记保证两波不重复完整 prompt。none 模式不共享首个完整 block；high 模式共享指定
前缀，并记录真实公共前缀长度。output=64/256 使用相同 prompt token stream。

完整数组写入忽略的 `runs/workloads/h2_nanovllm/requests.json`；
[`h2_workload_manifest.json`](../results/h_nanovllm/h2_workload_manifest.json) 保存
tokenizer、语料、生成器、配置和每个 case 的哈希。

[`h2_prefix_probe.json`](../results/h_nanovllm/h2_prefix_probe.json) 直接使用固定上游的
Sequence、BlockManager，分别执行 prefix 开/关，共 48 组检查。先批量 admission，再
发布完整 block hash，最后释放该波所有引用；warm 波保留 cold 波缓存元数据。

| workload | cold 每请求命中 | warm 每请求命中（开启 prefix） | warm 输入 token 命中率 |
|---|---:|---:|---:|
| 128-token prompt，high | 0 | 0 | 0% |
| 512-token prompt，high | 0 | 256 | 50% |
| none prefix，任一长度 | 0 | 0 | 0% |

关闭 prefix 的 adapter 仅令 `can_allocate` 返回 0 或容量不足，保留分配、hash 发布和
引用计数路径。GPU 挂接与跨路径 logits 对照由下面的 preflight 单独验收。
128-block pool 是元数据测试容量，不是 GPU 容量测量。CPU probe 不执行 attention、
Decode 或生成，也不据此推导速度与显存收益。

## 首轮 GPU 计划

先用四个 case：`b1_p128_o64_none`、`b8_p128_o64_high`、`b8_p512_o64_none`、
`b8_p512_o64_high`。每个 case 比较三个 variant，各三个独立进程，共计划 36 个进程。
其余 fixture 暂不进入首轮，避免直接启动完整 216-process 矩阵。

| variant | enforce_eager | prefix cache | 用途 |
|---|---|---|---|
| eager_no_prefix | true | false | 共同对照 |
| graph_no_prefix | false | false | 单独测 CUDA Graph |
| eager_prefix | true | true | 单独测 Prefix Cache |

`enforce_eager` 只控制 ModelRunner 的 Graph 路径，Sampler 与部分层的编译行为仍一致。
[`h2_execution_plan.json`](../results/h_nanovllm/h2_execution_plan.json) 是 CPU 计划，
`gpu_execution_allowed=false`；它不会自动启动实验。

## GPU 入口与验收门槛

[`serving/h2_runtime.py`](../serving/h2_runtime.py) 提供两个显式 adapter：
固定物理 KV pool 的 ModelRunner 子类，以及仅关闭 prefix admission 的 BlockManager
子类。通过 LLMEngine 的构造入口临时挂接，构造后恢复原类；不修改上游 checkout。
这不是声称上游原生提供了 prefix 开关或固定 pool API。attention、采样、Graph 捕获、
hash 发布与引用计数均沿用 pinned upstream。

首轮共同预算：BF16、TP=1、max_model_len=1024、max_num_seqs=8、
max_num_batched_tokens=4096、32 个 256-token KV blocks。32 blocks 不是自动推算的
58-block H1 pool；三条路径必须实际创建相同形状的 tensor。cold batch=8/prompt=512/
output=64 最多需要 24 blocks。分配时保留至少 512 MiB 设备空闲空间，运行后再次检查；
Graph 额外占用由实测 allocated memory 报告，不能以缩小 pool 掩盖。

1. H1 总流程、状态测试、attention、generation 四项结果均为 pass。
2. GPU 计时入口、adapter 正确性验证与共同 KV pool 的 preflight 均通过。固定所有
   variant 的实际 pool 大小，至少容纳 `batch * ceil((prompt + output - 1) / block_size)`
   个 blocks，覆盖整个 cold 波 Decode。模型、tokenizer、dtype、源码、pool、预算和请求
   必须一致；发生抢占或 OOM 的 case 不进入无抢占对照，也不以邻近成功点代填。
3. 保证 `max_num_seqs >= batch`、`max_num_batched_tokens >= batch * prompt`，否则
   首波分批执行后，后面的请求可能已命中刚发布的 prefix，不能称为 cold bulk control。
4. 编译、Graph capture 和 warmup 放在计时外并单独记录。每个测量 pair 开始前，在
   running/waiting 队列为空且引用全部释放时新建 BlockManager，复用相同物理 KV
   tensor，避免 warmup 或上一 pair 污染 cold 波。cold 与 warm 之间不清 cache。
5. 每进程五个 cold/warm pair，固定输出预算并设 ignore_eos=true。记录输出 token
   哈希；采样输出不用于质量评估，也不要求不同数值路径 bitwise 一致。

Preflight 在三个独立 GPU 进程中，对同一 `b8_p512_o64_high` case 的 cold/warm 各执行
四个 teacher-forced 生成步骤（Prefill + 三步 Decode）。每步将 BF16 全词表 logits
转为 FP32 保存和比较（不表示使用 FP32 执行了模型），
对 Graph/Prefix 路径分别与 eager_no_prefix 对照。事前阈值为 atol=0.05、rtol=0.02，
且 relative RMSE ≤0.005；top-1 一致率只作诊断。这验证同输入下 adapter/Graph 的跨路径
等价性，不是与 Transformers 的全模型参考比较。然后执行完整 64-token 请求检查实际
pool、命中、输出预算和释放状态。通过后生成绑定源码、模型、环境、配置的
`preflight.json`；任一相关文件变化会使正式运行 gate 失效。

### Prefix Cache 严格 gate 失败与定位（2026-10-23）

原始失败文件保留在 `results/h_nanovllm/h2_pilot/`，不与后续 Graph 结果混合。
原生 BF16 路径中 Graph 与 eager 的八组 logits 精确一致；Prefix cold 也精确一致，
但 Prefix warm 四步的最大绝对差异为 0.28125 / 0.25 / 0.25 / 0.3125，relative RMSE
为 1.44%～2.02%，超过事前阈值。该 probe 的 top-1 全部一致，但不能替代 logits 验收。

[`diagnose_h2_prefix.py`](../serving/diagnose_h2_prefix.py) 进一步执行受控诊断，证据为
[`prefix_numerical_diagnostic.json`](../results/h_nanovllm/h2_pilot/prefix_numerical_diagnostic.json)：

- 同一路径 A/A 全词表 logits 精确一致；物理 KV tensor 先填充 NaN，生成结果仍全有限。
- 28 层 warm 重算 prefix 与 cold 保存值精确一致；28 层复用 prefix 也精确一致。
- 第一层最后位置 Q/K/V 精确一致，attention 输出首次出现 0.001953125 的差异。
- **只替换 attention** 为 FP32 SDPA math 参考，仍保留 BF16 权重、线性层、norm 和
  原页式 KV 写入/读取；两路径在 cold/warm 全部八步的最终 logits 差异均归零。

这支持差异来自本 workload 的 BF16 contiguous/paged attention 数值路径，并经深层
模型累积；不是已证实的缓存页读错，也不能据此宣布原生路径通过严格等价验收。
FP32 intervention 是诊断，绝不混入 BF16 性能统计。PyTorch 对不同切片/批计算路径
不保证浮点结果完全一致，见[官方数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。
不修改原阈值，不发布 Prefix Cache 性能排名。后续已通过统一 paged Prefill 路径
消除本验证范围的差异，使用新策略 gate 和新目录；不是放宽误差预算，详见上方链接。

### Graph 的 batch=1 边界与 batch=8 合格范围

四个 case 的全部 32 个张量对照见
[`all_case_numerical_diagnostic.json`](../results/h_nanovllm/h2_graph_pilot/all_case_numerical_diagnostic.json)。
batch=1 的 Prefill 精确一致，但 Decode 最大差异达到 0.25390625；cold 第三个 Decode
步骤出现 top-1 分叉。该点在原四 case 预检时尚未定位，不能套用 Prefix 的 FP32-attention 诊断结论，
也不能用邻近 batch=8 的成功点填补。原四 case gate 仍为 fail，正式计时未启动。

batch=8 的三个 case 共 24 个 cold/warm/step 张量对照全部精确一致，因此独立冻结
`h2_graph_b8.json`，在新来源哈希/配置下重做 preflight，只有其通过才放行 18 个进程。
scope、allowed_variants、validated_case_ids 写入 gate，不能借 batch=8 Graph-only pass
放行 batch=1 或 Prefix Cache。该范围只回答“本机 batch=8 时 Graph 的收益”。

18/18 运行及后置审计已完成：1,440 个请求、92,160 个输出 tokens，720 对采样输出
哈希全部一致；重新聚合每请求计时与 summary 精确一致。正式吞吐倍率为 6.21～8.51×，
峰值 allocated 增加 8.144 MiB；这是原生 BF16 Graph-only 对照，不包含 FP32 诊断数据。

## 计时定义与边界

每进程先执行一个完整 cold/warm pair，覆盖该 workload 的 Prefill、Decode 和 prefix
路径，排除在正式测量外。初始化（包含上游 model warmup）、Graph capture 和额外
workload warmup 分别保存耗时。测量窗口前后检查 Dynamo 编译计数；发生新编译即失败。

这是一组**离线批请求的主机可见时延**，不是在线服务 SLO 或纯 GPU kernel 时延：

- 一波请求有共同起点，起点在 enqueue 前；每步使用原生 scheduler/run/postprocess。
  Sampler 返回的 `.tolist()` 已等待 GPU token 可见，取 postprocess 完成时间戳。
- TTFT 是起点至首 token，E2E 是起点至最后 token；TPOT 是每请求
  `(E2E-TTFT)/(output_tokens-1)`，不是单步 inter-token latency 的 p99。
- 每 phase 的分位数来自该进程 5 个 pair 的请求样本；同批请求有相关性，不能视为
  40 个独立进程。吞吐为总 output tokens 除以该 phase 五波 wall time 之和。
- 包含 enqueue、调度、输入准备、模型计算、采样、token 拷贝和 postprocess；排除
  初始化、warmup、pair 间 cache metadata reset、输出 hash 和磁盘写入。
- 内存采用各波 `max_memory_allocated` 最大值，包含模型、固定 KV pool 和 Graph
  常驻 tensor；不等于 nvidia-smi 或 PyTorch reserved memory。

完整每请求计时、输出 token hash、cache 命中、编译计数与释放状态写入 raw JSON。
首轮按 run 轮转三个 variant 的执行顺序，减轻所有 Graph 都在热机后执行的顺序偏差；
仍只适用于本机电源/温度条件，不能推出生产环境普遍收益。

## 汇总器 raw-result 契约

[`scripts/summarize_h2_nanovllm.py`](../scripts/summarize_h2_nanovllm.py) 已完成输入校验和
汇总逻辑。单元测试数值是人工 fixture，没有写成正式实验结果。

| 字段 | 要求 |
|---|---|
| 顶层 | schema_version=1、stage=H2、status=pass、evidence_class=formal_gpu、case_id、variant、run_id |
| provenance | upstream_commit、model_revision、model_sha256、tokenizer_sha256、source_sha256、environment |
| engine | 两个 bool 开关、dtype=bfloat16、实际 num_kvcache_blocks、kvcache_block_size、max_num_batched_tokens、max_num_seqs、max_model_len，以及其余共同参数 |
| workload | sha256、batch_size、prompt_length、output_length、prefix_mode、measured_pairs、sampling、cache_reset_policy=fresh_block_manager_per_pair |
| phases.cold / phases.warm | TTFT/TPOT/E2E 的 p50/p95/p99 毫秒、output_tokens_per_second、peak_allocated_bytes、requests_completed、prompt_tokens、output_tokens、cached_prompt_tokens |

每 variant 恰有 run1/run2/run3，跨进程 invariants 一致；跨 variant 只允许 Graph/Prefix
两个开关变化。汇总器拒绝 CPU evidence、缺失请求、输出预算不同、NaN、分位数倒置、
cold/disabled/短 prompt 中的异常命中、缺失对照，以及模型或 KV pool 偷换。

采用独立进程统计量的中位数，cold/warm 分开比较，不合并成全局尾延迟分位数。
GPU 入口产生 raw JSON 后才能生成正式 summary。

## GPU 首轮复现

```bash
runs/envs/nano-vllm/bin/python scripts/run_h2_nanovllm.py --preflight-only
runs/envs/nano-vllm/bin/python scripts/run_h2_nanovllm.py --resume
```

以上原三变体配置目前会在 Prefix 数值 gate 失败；不应反复重试或跳过检查。
原四 case 的 `h2_graph_pilot.json` 也不能通过全部数值门槛。
历史 batch=8 Graph-only 首轮复现使用：

```bash
runs/envs/nano-vllm/bin/python scripts/run_h2_nanovllm.py --config config/serving/h2_graph_b8.json
# 中断后仅续跑同一 gate 的未完成结果
runs/envs/nano-vllm/bin/python scripts/run_h2_nanovllm.py --config config/serving/h2_graph_b8.json --resume
.venv/bin/python scripts/audit_h2_nanovllm.py
```

新策略的正确性复验使用 `scripts/run_h2_corrected_validation.py`，见
[`h2_numerical_fixes.md`](h2_numerical_fixes.md)。它不重新启动性能矩阵，也不修改历史
native gate。新策略的性能入口为：

```bash
runs/envs/nano-vllm/bin/python scripts/run_h2_corrected_pilot.py
runs/envs/nano-vllm/bin/python scripts/run_h2_corrected_pilot.py --resume
runs/envs/nano-vllm/bin/python scripts/audit_h2_corrected_pilot.py
```

该入口对全部三个变体调用同一 `create_corrected_engine`，不能只给某个变体挂接修正。
配置固定数值 gate 的 SHA256；每个进程重新核对模型、环境、源码、全部 64 步 gate
覆盖与 workload hash。性能入口本身另存源码哈希，不修改已绑定旧证据的代码。
结果仅进入 `h2_corrected_pilot/`，不覆盖或混用历史数据。完整复现和适用范围见
[`新策略性能记录`](../results/h_nanovllm/h2_corrected_pilot/README.md)。

入口有单实例锁，子进程单卡串行执行；失败立即停止并保留日志。`--resume` 只跳过与
当前 preflight hash 匹配且校验通过的结果，不混入旧版本 pass。完整 token fixture
仍包含 24 个 case，但本入口只允许四个 pilot case，不自动启动 216-process 全矩阵。
数值对照 tensor 放在忽略的 `runs/validation/h2_nanovllm/<pilot>/`；日志放在
`runs/logs/h_nanovllm/<pilot>/`；小体积结果放在 `results/h_nanovllm/<pilot>/`。

## CPU 复现

```bash
CUDA_VISIBLE_DEVICES='' runs/envs/nano-vllm/bin/python -m serving.h2_workloads
CUDA_VISIBLE_DEVICES='' runs/envs/nano-vllm/bin/python -m serving.h2_prefix_probe
.venv/bin/python -m unittest discover -s tests -p 'test_h2_preparation.py' -v
```

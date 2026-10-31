# H2 数值修正：batch=1 Graph 与 Prefix Cache

日期：2026-10-23。两项问题在显式新策略 `static_tables_paged_prefill_v1` 下通过复验；
不修改历史 native 失败 gate，也不把历史 batch=8 性能数字套用到新策略。

## 修正及证据

| 问题 | 原对照的额外差异 | 修正 |
|---|---|---|
| batch=1 Graph | eager 页表宽度随请求变化，Graph 固定 4 列，影响 attention 分块选择 | 两侧页表均固定为模型窗口容量 |
| Prefix warm | off/无命中使用 contiguous Prefill，on/命中使用 paged Prefill | 已分配 KV 的 Prefill 在两侧统一使用 paged 路径 |

保持原 BF16、权重、tokenizer、32-block 物理池与命中规则；不采用 FP32 attention
参考作正式修复，也不放宽原容差。Dynamo 缓存容量显式设为 64，触顶/编译异常直接
报错，防止额外形状和调试调用导致静默 fallback。

[`validation_v1/gate.json`](validation_v1/gate.json)：**pass**。
eager_no_prefix、graph_no_prefix、eager_prefix 三个独立 GPU 进程，共八个 case，
包括原四个 case 和四个新 seed / 长度，覆盖 batch=1/4/8、prompt=128/256/384/512/768。
每 case 的 cold/warm 各执行完整 64 步，每步比较 BF16 **全词表输出字节**，共
**2,048 个张量对照全部精确一致**，而非只检查 top-1 或最终文本。
[`validation_v1/audit.json`](validation_v1/audit.json) 已复核全部逐步对照、源码/权重哈希、
配置与历史原始结果；CPU 回归为 96 项中 94 项通过、2 项 CUDA/Triton 专项按环境设置跳过。

共执行 234 个 teacher-forced 请求、14,976 个输出 tokens。128/256-token 负对照
无 prefix 命中；512/384-token high warm 复用 256 tokens/request；768-token high
warm 复用 512 tokens/request。全部 logits 有限，各波结束后请求队列、used blocks
与非零引用计数均为 0；三个进程均开启 fail-on-recompile-limit，无静默回退。

这仅是新策略的正确性验收，不是 Transformers 全模型参考、模型语义质量评估，
也不含新的 TTFT/TPOT/吞吐 benchmark。新的性能矩阵需统一应用该策略，并生成新的
来源信息和结果目录；不得与历史原生路径合并排名。
这轮性能矩阵已在独立的 [`h2_corrected_pilot/`](../h2_corrected_pilot/README.md) 完成，36/36 进程与审计通过；
此目录仍只保存数值验收与定位证据。

## 定位记录

- [`shape_probe.json`](shape_probe.json)：原路径、静态页表、paged Prefill 的受控介入。
- [`graph_layer_trace.json`](graph_layer_trace.json)：逐层 debug 追踪；触发编译缓存上限，
  存在调试干扰，**不作为原问题的根因结论**。
- [`graph_compile_probe.json`](graph_compile_probe.json)：新进程禁止回退后，原页表仍
  失败，统一页表后八个对照精确一致。
- [`validation_v1/run.json`](validation_v1/run.json)：独立进程执行记录；同目录的三个
  variant JSON 保存完整逐步 logits 哈希、输入哈希、命中与资源释放状态。

入口：[`h2_corrected_runtime.py`](../../../serving/h2_corrected_runtime.py) 的
`create_corrected_engine` / `close_corrected_engine`。原生 harness 与上游 checkout
不改动，历史 [batch=8 审计](../h2_graph_b8/audit.json) 仍能校验原源码/结果。

完整定位过程、为何不能把编译回退误当根因、复现命令及验证 case 见
[`h2_numerical_fixes.md`](../../../docs/h2_numerical_fixes.md)。

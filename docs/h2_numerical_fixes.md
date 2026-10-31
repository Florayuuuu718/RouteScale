# H2 两项数值差异：定位、策略修正与复验

本文只处理原 batch=1 CUDA Graph、Prefix Cache 的数值门槛，不启动新的训练或性能矩阵，
不改动上游 checkout、权重、tokenizer、BF16 精度或原容差。
新策略为 `static_tables_paged_prefill_v1`，配置见
[`h2_corrected_validation.json`](../config/serving/h2_corrected_validation.json)。
最终执行状态与门槛分别记录在
[`run.json`](../results/h_nanovllm/h2_numerics/validation_v1/run.json)、
[`gate.json`](../results/h_nanovllm/h2_numerics/validation_v1/gate.json)。
**本次复验已完成且通过**：三独立进程、八个 case、2,048 组逐步全词表 logits 字节哈希
全部一致。历史失败仍保留；新策略没有性能排名。

## 为什么原对照不够单变量

| 问题 | 原对照隐藏的路径差异 | 新策略，两侧同时应用 |
|---|---|---|
| batch=1 eager / Graph Decode | eager 页表随实际 block 数变化，Graph 页表固定 4 列 | `prepare_block_tables` 均补齐到模型窗口容量，当前为 4 列 |
| Prefix off / on Prefill | 无命中时是 contiguous attention，有命中时是 paged attention | 已分配 KV 的 Prefill 始终走 paged attention；命中开关只控制复用 |

页表容量不等于实际有效 token 长度。补齐保留有效 page IDs 和序列长度，额外列不增加
请求长度，也不允许读取未计算的 KV。cold 前将整个物理 KV tensor 填为 NaN，随后检查
所有 logits 有限、命中量正确、资源释放，用于发现错误读取。

FlashAttention 2.8.3 的实现会从页表宽度推导 KV 容量，供 Decode 自动分块选择使用；
Prefill 的 paged 标志也会切换内核族。原先因页表或命中而变化的计算路径，可能在
BF16 下产生不同舍入。这里统一实现路径，而非降低数值标准。参见
[固定版本上游实现](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/flash_api.cpp)。

这不是“原生 FlashAttention 算错了”的证明，也不代表新策略的模型质量更好；修正目的
是让 Graph / Prefix 开关不再同时偷偷改变另一项计算条件。

## 定位过程中保留的证据

1. [shape_probe.json](../results/h_nanovllm/h2_numerics/shape_probe.json)：同引擎只统一
   paged Prefill 后，Prefix 八个张量对照误差归零；单独统一页表不足以使当时的 Graph
   调试运行通过。
2. [graph_layer_trace.json](../results/h_nanovllm/h2_numerics/graph_layer_trace.json)：增加
   逐层 debug copy，并在同进程切换编译条件时，触发了 Dynamo 的 8 项重编译缓存上限，
   一侧 fallback，而捕获图仍回放已编译计算。该追踪发现 norm 分叉，但有调试干扰，
   **不将其作为原失败的根因证据**，也不把 debug 路径用于性能计时。
3. [graph_compile_probe.json](../results/h_nanovllm/h2_numerics/graph_compile_probe.json)：
   新进程设置编译容量 64，并在触顶时直接报错；原宽度策略仍失败（最大差异
   0.25390625），仅统一页表后，cold/warm 八个对照全部精确一致。额外对齐 capture
   时默认 dtype/device 并非必要条件。

因此新 serving 策略也明确固定 `recompile_limit=64`、
`fail_on_recompile_limit_hit=True`、`suppress_errors=False`：它们用于防止后续形状扩展
时静默回退污染对照，不是通过换精度来掩盖误差。这些内部设置依赖已锁定的 PyTorch
2.8 环境，仅限独立 serving 进程，不修改训练环境。

## 可复用入口

[`h2_numerical_policy.py`](../serving/h2_numerical_policy.py) 只提供实例级 shape adapter；
[`h2_corrected_runtime.py`](../serving/h2_corrected_runtime.py) 暴露成对的
`create_corrected_engine` / `close_corrected_engine`，用于新策略的构造与清理。
后者恢复绑定方法并释放 restorer 闭包，再退出上游引擎，避免保留模型/Graph 引用。
原 `h2_runtime.py` 不修改，因此历史 benchmark 的源码哈希仍可审计。

不要在旧结果目录覆盖 gate，也不要只给某一个 variant 应用新策略；新的性能矩阵应
使用统一策略、重新生成 provenance / gate，并写入独立目录，不能与历史原生测量合并。

## 完整 64 步验证契约

[`verify_h2_corrected.py`](../serving/verify_h2_corrected.py) 不做采样，不记录性能，使用
各请求 prompt 尾部反向的 64 个 token 作固定 teacher-forced 续写。每一步将 BF16 的
**全词表输出字节**计算 SHA256；三条路径在独立进程运行，不保留大体积中间张量。
相同形状/dtype 下字节哈希完全一致，比原 atol=0.05、rtol=0.02、relative RMSE≤0.005
更严格；不能用 top-1 或最终生成文本相同代替。
三条路径共执行 234 个固定续写请求、14,976 个输出 tokens；全体有限性、命中和资源
回收检查通过。GPU 执行完成后已释放。

覆盖原四 case，以及未参与原先定位的新 seed / 长度：

| batch | prompt | output | prefix | 输入 |
|---:|---:|---:|---|---|
| 1 | 128 | 64 | none | 原 fixture |
| 8 | 128 | 64 | high | 原 fixture，短于 block 的负对照 |
| 8 | 512 | 64 | none | 原 fixture |
| 8 | 512 | 64 | high | 原 fixture，warm 复用 256 tokens/request |
| 1 | 256 | 64 | none | 新 seed=20261024，完整页及 Decode 跨页 |
| 1 | 512 | 64 | high | 新 seed=20261025，单请求 prefix 命中 |
| 8 | 384 | 64 | high | 新 seed=20261026，warm 复用 256 tokens/request |
| 4 | 768 | 64 | high | 新 seed=20261027，warm 复用 512 tokens/request |

共同引擎预算仍为 4096 batched tokens、8 sequences、1024 窗口、32×256 KV blocks；
768-token case 使用 batch=4，保证 cold 一波整体 admission，不引入 chunked Prefill
或抢占。每个 case 均检查 64 步、全部 logits 有限、实际命中、最终队列/页引用回收。

```bash
runs/envs/nano-vllm/bin/python scripts/run_h2_corrected_validation.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -p 'test_h2_*.py' -v
```

runner 有单实例锁，三进程串行，失败即停。已存在结果目录不会覆盖；另一次复验须
复制配置并设置新的 `results_dir`。`gate.json` 只放行该新策略及实际验证 case；它不
追认历史 native gate，不提供新策略吞吐排名，也不是与 Transformers 的全模型参考
比较或语义质量评价。

# RouteScale G 阶段原生推理实验协议

## 1. 目标与边界

G3 只回答 RouteScale Dense 模型在单卡上的原生推理问题：完整上下文重算与连续
KV Cache 分别怎样影响 TTFT、逐 token 延迟、端到端延迟、吞吐和显存。G3 不加入
Tensor Parallel、连续批处理、Prefix Cache 或在线请求调度；这些变量留给 H/I 阶段。

正式结果必须使用可访问的 CUDA GPU。CPU 仅用于验证代码、结果 schema 和汇总逻辑，
不得进入性能结论。

## 2. 固定对象

- 推理产物：`runs/checkpoints/g_serving/dense_baseline/model.pt`；
- 数据：`data/tinystories_full/val.bin`；
- dtype：BF16；
- 后端：`naive` 与 `kv_cache`；
- batch：1、8、32；
- prompt：128、512 tokens；
- output：64、256 tokens；
- Decode 输入：固定的 TinyStories validation continuation；模型仍执行 argmax，预测
  token 另存哈希，但下一步统一喂入固定 token，避免 BF16 近似差异改变后续 workload；
- prefix workload：无共享前缀、75% 共享前缀；
- 每个 variant：2 次预热、5 次测量、3 个顺序启动的独立进程。

完整矩阵有 48 个 variant、每个 3 次独立运行，共 144 个进程。所有进程固定到同一张
GPU 并顺序执行，避免其他 GPU 或并发任务引入差异。

## 3. 测量边界

模型和 artifact 加载不进入计时。单次请求批次记录：

- 首个生成 token 的 TTFT；
- 后续 token 的逐步延迟与 TPOT p50/p95/p99；
- 同步后的请求端到端 wall time p50/p95/p99；
- request throughput、output token throughput 和 decode throughput；
- peak allocated/reserved CUDA memory；
- cache 分配字节数和因上下文滑窗发生的 cache rebuild 次数；
- workload、输出 token、artifact 和源码 SHA256。
- 模型预测 token 哈希；真实 greedy 的数值分叉单独作为正确性诊断，不混入计时矩阵。

CUDA kernel 时间使用逐 token CUDA Events；请求 wall time 在测量前后各同步一次。三个
独立进程先各自产生统计量，最终结果使用三个 run median 的中位数。

## 4. 上下文边界

当前模型使用 learned absolute position embedding，`block_size=512`。因此
`prompt_length + output_length > 512` 时，窗口每次左移都会改变保留 token 的位置编号，
旧 KV 不能直接复用。实现必须重建 cache，结果中标记 `context_overflow=true`。

这类 workload 是滑窗压力测试，必须与 128-token prompt 的正常增量 Decode 分开解释；
不能把重建后的低收益归因于 KV Cache 机制本身。

## 5. 执行

先确认只有目标卡可见，然后运行：

```bash
python scripts/run_g3_native_matrix.py --gpu=0 --dry-run
python scripts/run_g3_native_matrix.py --gpu=0
```

中断后可跳过已通过的正式结果：

```bash
python scripts/run_g3_native_matrix.py --gpu=0 --resume
```

原始 JSON 写入 `results/g_serving/g3_native/raw/`，汇总写入
`results/g_serving/g3_native/summary.json`。

## 6. 验收

1. 每个 variant 恰有 `run1/run2/run3`，环境、artifact、源码和 workload 哈希一致；
2. 每次测量输出确定，naive/KV 对照使用完全相同的 prompt；
3. 所有指标来自无 Profiler 的 CUDA 运行；
4. 正常上下文与滑窗重建场景分开报告；
5. OOM 或执行失败保留配置和失败阶段，不以邻近成功点代填。


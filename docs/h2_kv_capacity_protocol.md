# H2：KV 分配容量、内部浪费与连续空洞

配置：[h2_kv_capacity.json](../config/serving/h2_kv_capacity.json)。
结果：[h2_kv_capacity/](../results/h_nanovllm/h2_kv_capacity/)。
这是一轮分配/数值实验，不启动性能矩阵，也不改教学大纲。

## 共同预算与三个基线

沿用固定 Qwen3-0.6B、BF16、TP=1、1024-token 窗口。真实 KV tensor 的形状为
`[2, 28, 32, 256, 8, 128]`：每个 KV token 为 112 KiB，每页 28 MiB，总计
896 MiB，逻辑总量 8192 token 槽。Prefix sharing 关闭，不把共享前缀混入容量收益。

| 分配策略 | 预留方式 | 实现与证据边界 |
|---|---|---|
| static_window | 每请求预留完整 1024-token 连续窗口 | 新 adapter 管理上游真实 block/refcount；GPU 仍用 paged attention |
| contiguous_exact | 按当前 KV 长度，在 token 粒度 arena 中 first-fit | 仅 CPU 显式参考，无 GPU attention 或耗时结论 |
| paged | 按当前 KV 长度，以 256-token 页增长 | pinned upstream BlockManager，关闭 Prefix admission |

GPU 两侧只改变逻辑预留/页映射，不切换 attention kernel，不修改权重、dtype 或 KV
tensor 布局。它不是“原生连续 attention 对比原生分页 attention”。固定窗口的预留
空闲不是外部碎片；按需连续分配也可能比分页更省内部空间，必须保留这个负对照。

## 分配器重放

1. 当前活跃 KV 长度固定为 128 / 256 / 257 / 512 / 768 / 1024，逐个申请至第一次
   拒绝，不释放中间请求。记录 live / page-tail / future-reserved / free token 槽，
   再释放全部请求。这里测的是 KV 存储容量，不是模型 batch 或服务并发。
2. 连续空洞：在两侧都可容纳的相同初态申请 16 个 512-token 请求，释放偶数编号，
   再申请一个 768-token 请求。连续 first-fit 总空闲 4096、最大空洞 512，不能满足
   768；分页能组合散布的三页。固定窗口策略无法容纳这个初态，不伪造一组可比结果。
3. 增长：四个 256-token 请求中，把第一个增长至 257。连续参考允许先申请新空间、
   再复制旧 KV 后释放，因此记录所需复制量；分页保留旧页并追加一页。复制量是
   逻辑工作量，不是测得的 GPU copy 带宽或延迟。连续参考不做全池 compaction。

每次状态变化检查资源守恒、页归属唯一、真实 refcount、全部释放。页尾浪费单独记录，
不能把 arena 的未分配空闲都算成“碎片”。

## 真实 GPU 地址与模型检查

两个独立模型进程，顺序执行 static_window、paged，各有相同 896 MiB tensor：

- 重放各自六个容量点，对所有已拥有页的全部元素写入/读回 BF16 owner 哨兵；空闲页
  必须仍为 NaN。分页再检查一次非连续空洞复用。前后 tensor 指针及 PyTorch allocated
  字节必须不变。此处没有执行容量上限数量的模型请求，不能宣称测得最大服务并发。
- 完整模型使用相同八个 128-token prompt，各执行 132 步 teacher forcing。生成后
  最后一个 token 尚未写入 KV，所以最终活跃缓存为 `128+132-1=259`，而不是 260。
  记录 KV 长度 128 / 256 / 257 / 259 的实际页表与已用页数，跨越真实 Decode 页边界。
- KV tensor 先全部填 NaN，全部 logits 与检查点有效 KV 必须有限。两侧固定同一
  `static_tables_paged_prefill_v1` 和 Prefill BF16 tile512 控制，Decode Graph 不变。
- 静态预留页单独保管，`seq.block_table` 只暴露当前 KV 页。上游 `prepare_decode`
  使用 `block_table[-1]` 确定写入地址，暴露未来预留页会造成错误。
- 两侧完整 BF16 logits 保存到忽略的 `runs/validation/h2_kv_capacity/`，独立审计进程
  核对 SHA256、重新比较 1056 个全词表向量，要求全部 exact，不放宽阈值。

数值与地址检查不记录 TTFT / TPOT / tokens/s；初始化耗时只作诊断。模型执行固定为
batch=8，32 个逻辑小请求可分配不代表它们已同时完成 Prefill / Decode。

## 入口与输出位置

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_h2_kv_capacity
runs/envs/nano-vllm/bin/python scripts/run_h2_kv_capacity.py
runs/envs/nano-vllm/bin/python scripts/audit_h2_kv_history.py --check-only
```

首个 runner 仅允许新结果目录，使用公共 `runs/tmp/h2_runner.lock` 串行运行三个子进程。
默认配置有历史结果后不重复覆盖；`--check-only` 只读核对现有源码、模型、环境、
数值张量哈希和历史记录。完整张量比较在首轮 runner 的 audit 子进程执行；不要对
已冻结结果直接重写 audit。源码、配置、模型、环境与上轮合格 gate 都被绑定；旧失败、
既有 18/36/18 性能矩阵及大纲不改。

| 内容 | 目录 |
|---|---|
| 状态、GPU 小型 JSON、分配重放、审计 | `results/h_nanovllm/h2_kv_capacity/` |
| 两侧完整 BF16 logits | `runs/validation/h2_kv_capacity/h2_kv_capacity/` |
| GPU 子进程日志 | `runs/logs/h_nanovllm/h2_kv_capacity/` |

后续若要测 KV 压力下的吞吐/服务容量，需要另定接入上限、增长预算、抢占/换出/
compaction 策略，并重新通过对应数值门禁；本轮不授权为这些工作签发性能 gate。

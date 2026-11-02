# H3 性能 v5：FP32 数学误差预算与统一 BF16 累积策略

2026-10-24，在任何 v5 资格验证或计时前冻结。
配置：[h3_performance_v5.json](../config/serving/h3_performance_v5.json)。
本轮将 [v3](h3_performance_v3_protocol.md)预先定义的 FP32 数学误差门槛，与
[v4](h3_performance_v4_protocol.md)验证有效的共同 BF16 累积策略组合；被测模型、
输入、权重、输出 dtype、四 case、三路径、两种计时范围、预热、pilot CV 和正式
36 进程矩阵仍按 [v1](h3_performance_protocol.md)执行。

## 数值依据及历史限制

默认 BF16 GEMM 归约许可开启时，v3 的 FP32 数学参考下 native/page_growth 和
paged/batch_short 超过 0.5% relative RMSE，未启动性能测量。
[单开关因果诊断](../results/h_nanovllm/h3_performance_v3/reduction_policy_diagnostic.json)
关闭中间累积降精度许可后，两处误差分别降至 0.334612% / 0.430108%。

v4 在此策略下重新生成三组参考，发现三条目标路径全部满足 v3 的 FP32 数学预算，
也全部满足 batched BF16 参考，但相对单请求 BF16 参考的 batch_short 有超预算点，
最大 0.570047%。[v4 gate](../results/h_nanovllm/h3_performance_v4/gate.json)仍为 fail，
正式计时仍为零。不能把“各自相对数学参考误差≤0.5%”当成“任意两个 BF16
舍入路径互相差异也≤0.5%”；误差预算没有这种传递性。

v5 以 v3 已定义的数学基准为性能执行的唯一数值资格条件，不继续选择新的参考、
数据窗口、阈值或有利 case。BF16 互比失败作为持续披露的限制，v1–v4 均不改写。
本轮只证明相对该数学基准的预算，不宣称通用 BF16 bitwise 或 pairwise 等价。

## 固定合同

- 三条目标：native_kv、paged_eager、paged_graph，全部 BF16 输入／权重／输出，
  关闭 autocast 与 Prefix，TP=1；共同设置
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False`，在模型
  初始化、warmup 与 Graph capture 之前生效，实际值写入 provenance。
- 数学参考：同 artifact 先舍入 BF16，核对三侧相同的 BF16 参数哈希，再提升为
  FP32，逐请求 full recompute、math SDPA、关闭 TF32。参考计算不参与性能计时。
- 新进程重建全部参考与目标，不用 v4 的 pass 子项冒充 v5 的新 gate。2592 个
  BF16 全步全词表目标向量全部检查 finite、pointwise `atol=0.125, rtol=0.02`，
  每向量 relative RMSE≤0.005；864 个 eager/Graph 向量还须 bitwise exact。
- case、seed、prompt 与 continuation 数组完全沿用 v1。不得更换 token 或 shape
  来通过门槛。native 与分页 KV 的物理分配差异如实报告。
- gate 通过后，六个独立 pilot 均达到原 CV≤15%、输出、页增长、回收和显存门槛，
  方可运行 4×3×3=36 个正式进程。两种测量范围分别预热三轮、测五轮；三次独立
  进程轮转路径顺序，报告中位数和离散程度。
- `model_execution` 为 CUDA Event 模型区间，`replay_e2e` 包含共同控制 harness、
  argmax 与 CPU 交接。结果属于固定 token 回放，不是自然生成、上游公共文本 API、
  HTTP、在线 SLO 或最大容量证明。不将 FP32 参考或旧 G3 数字混入对照。
- 默认累积策略的失败与这一显式策略的计时分别报告；不能用本轮声称原默认策略通过。

结果、请求、大张量、日志、编译缓存使用各自 `h3_performance_v5` 新目录。
记录冻结源码／协议／配置副本和全部旧 G/H JSON 的起始哈希，独立重算与核对后交付。

```bash
.venv/bin/python scripts/run_h3_performance.py --config config/serving/h3_performance_v5.json
runs/envs/nano-vllm/bin/python scripts/audit_h3_performance.py --config config/serving/h3_performance_v5.json --check-only
```

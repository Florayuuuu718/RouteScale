# I2 本机单卡受控消融协议 v1

本协议冻结在新 GPU 消融之前。使用 I1 已验收的上游、Qwen3-0.6B revision、
独立 Torch 2.9.1+cu129 环境和本机 RTX 5060 Laptop 8 GiB。
原 I1 补丁、配置、源码树和证据保持不变；I2 扩展单独位于 `patches/minisgl-i2`。
不启动付费资源。

## 固定输入和对照

`config/serving/i2_single_gpu.json` 固定 TP1、BF16、FlashInfer fa2、page-size=1、
2048 个真实 KV 页、窗口 1024、4 个执行 table；比较 batch=1/3、输入 512 token、
输出 64 token。原始 token ID 工作负载在本机 `runs/workloads/i2_single_gpu_v1`，
每次实验复制到结果目录，记录文件、输入、builder 和语料哈希。

零共享组的各个请求第一个 token 就不同；共享组在精确的 384 token 后分叉。
每个 case 都从无 prefix-cache 条目的 cold wave 开始，接着运行不同请求的 warm wave。
Naive 的 warm 命中必须为 0；Radix 的共享 warm 每请求必须命中 384 token。
cold 表示 prefix cache 起始为空，允许同一 cold cohort 内调度带来的真实共享；
它不表示 CUDA/JIT、模型权重或 OS 文件缓存为空。

七个配置逐项相对 `radix_full` 改变一个变量：Naive/Radix，完整 prefill/256/128 chunk，
Graph bucket `[1,2,4]` / `[1,3]`，以及 Overlap。完整 prefill 预算 4096，大于整个 cohort
的 1536-token prompt；Graph 的真实 capture/replay/padding 都记录，不能仅凭开关标签推断。
batch=3 在 `[1,2,4]` 下实际 pad 到 4，而 `[1,3]` 不 padding。

## 数值门槛

先在独立进程构建 HF eager 参考：同一 checkpoint 先装入 BF16 权重，再升为 FP32 运算，
禁用 TF32。目标仍使用原生 BF16 权重/激活，所有目标共同禁用 BF16 GEMM reduced-precision
reduction。这是明确的 I2 数值策略，不回写 I1，也不混用旧 H 阶段性能。

每条请求固定使用输入最后 64 token 的倒序作为共同 continuation。原生 sampler 的
argmax 仍执行，控制器将返回 token 替换为已驻留 GPU 的共同 token；参考和目标都消费
同一逐步输入流。它是完整模型的固定 token 回放，不是生成质量或在线自然输出性能结论。

每配置、每个 cold/warm 请求的全部 64 步、全词表 151936 维都检查，
每配置 1024 个 vector、七配置合计 7168 个 vector。门槛在执行前固定为
`atol=0.125, rtol=0.02, relative_RMSE<=0.005`，全部值有限且每元素均通过。
记录 top-1 一致率，但不用采样文本相同代替完整数值检查。门槛失败保留原结果；不得放宽门槛后
把旧实验改成通过。KV cache 在冷数值运行前填 NaN，检查未初始化/错误页读取。

数值观测采用 GPU clone，整波结束后才复制到 CPU；禁止每步 `.cpu()` / finite `.item()`
把 Overlap 串行化。完整参考/目标 tensor 留在忽略的 `runs/validation`，小型误差、哈希、
调用和释放证据进 Git。独立审计重新载入 tensor，不只信任 worker 的 passed。

## 生命周期与 Overlap

Overlap 保留一轮在途 forward。长度终态按 host 已提交 token 数计算，不能按提前推进的
device_len；EOS/abort 释放前同步相关在途工作，迟到结果丢弃。已退休 UID 只保留到对应
在途 batch 被处理，不能积累永久 tombstone。共享 tokenizer 的有序取消确认沿用 I1；
独立 tokenizer 和 TP2/4 仍不支持。

每个配置再启动真实 HTTP 服务，使用同样的页池、4 个执行槽和该配置的 cache/chunk/graph/
overlap，验证 EOS、length、混合长短请求、decode 断开、排队取消、deadline、后续 probe，
以及 frontend/detokenizer/table/KV 引用释放和 10 秒关闭。排队取消先观察到四个 blocker
实际入模；不能用 HTTP 发起顺序假定排队。该链路观测关闭 I1 的同步 finite sampler hook，
保留真实异步执行；数值检查由独立门槛承担。每请求只允许一个客户端终态。

## 计时及结论边界

数值门槛及七配置生命周期门槛全部通过后才能计时。正式矩阵为七配置、三个独立进程：
21 次，每进程三个 warmup 和五个 measured cold/warm 配对，包含全部四个 workload case。
第一轮按配置正序、第二轮逆序、第三轮轮转执行；同机全局 GPU 锁串行执行。
记录温度、时钟、功耗、P-state、显存、实际 batch 查询数和真实 Graph replay 数。

主要 wall 边界从 Scheduler 接收 cohort 到全部 host 终态与异步 GPU 工作结束，
包含请求调度、prefix 查询、metadata、完整模型、sampler、固定 token 控制和 host 结果处理；
排除启动、编译、tokenizer、HTTP、日志落盘和结束后的资源审计。
首结果指标是 Scheduler first-result，不称作网络 TTFT；不从 SSE frame 推导 token 间隔。
完整 batch 轨迹先在内存收集、计时结束后落盘，计时路径无逐步 finite/CPU tensor 复制。

每次波结束独立检查 table 槽与 prefix/free KV 页索引无重复、无缺失及所有活跃锁释放。
允许可驱逐 prefix 条目和 CUDA reserved 保留。报告三个进程的均值、范围和配对比值范围，
不把五个进程内重复当作五个独立样本，不只挑最佳值，也不从小样本宣布精确稳定排名。
I3 后续单独使用真实 HTTP 到达负载验证在线延迟与 goodput。

## 复现

```bash
runs/envs/mini-sglang/bin/python scripts/prepare_i2_minisglang.py
runs/envs/mini-sglang/bin/python -m serving.i2_workloads
runs/envs/mini-sglang/bin/python -m unittest discover -s tests -p test_i2_single_gpu.py -v
runs/envs/mini-sglang/bin/python scripts/run_i2_single_gpu.py --attempt fresh001 --phase all
runs/envs/mini-sglang/bin/python scripts/audit_i2_single_gpu.py --directory results/i_minisglang/i2_single_gpu_v1/fresh001
```

每个新 attempt 冻结源码、补丁、模型、配置和工作负载。进程失败或源码变更后使用新 attempt。
部分执行不等于 I2 完成；仅 `local_single_gpu_complete=true` 表示本机矩阵验收，
TP2/4 保持后期同机集中租卡待办。Attention backend 仅在实际支持时另开单变量比较，
本矩阵不把 fallback 或未测 backend 计为已比较。

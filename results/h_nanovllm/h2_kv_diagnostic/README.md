# H2：充裕预算负结果的后续诊断

状态：[run.json](run.json) 和[审计](audit.json)通过。三个独立 GPU 进程完成
48 条完整请求配对 trace、27 条 CUDA stream 探针；独立重读保存张量后，3072 个
全词表向量全部 exact。没有修改运行时，也没有重写[原配额实验](../h2_kv_pressure/README.md)。

结论：原来 0.931× 的整体吞吐退化在本轮平衡配对中未复现，不能据旧结果认定
分页存在固定 6.9% 惩罚。历史具体根因仍未唯一确认，不能直接改称“纯噪声”或
“已修复”。协议和复现见 [h2_kv_diagnostic_protocol.md](../../../docs/h2_kv_diagnostic_protocol.md)。

## 完整请求：同一模型进程内配对

沿用原执行函数、模型、896 MiB KV tensor、16 个输入、全部 64-token 输出、真实
sampler 与原计时边界。每区组采用 ABBA/BAAB 平衡顺序，并重置到同一 seed。
先两侧各完整预热一次；每进程四个区组，共三个独立进程。每区组四次的逐请求
采样哈希全部一致，所有队列/引用释放，计时前后编译计数均为 21。

| 独立进程 | 四个区组的分页/固定预留吞吐比 | 进程中位数 |
|---|---|---:|
| run1 | 1.042 / 1.037 / 0.986 / 0.993 | 1.015 |
| run2 | 1.009 / 0.996 / 1.027 / 1.025 | 1.017 |
| run3 | 1.009 / 1.035 / 1.073 / 1.098 | 1.054 |

三个进程中位数再取中位数为 **1.017×**。12 个区组范围为 0.986～1.098，9 个
高于 1、3 个低于 1；这些相关区组不能当成 12 个独立进程来作显著性判断。
这里不是把原 0.931× “修正”为 1.017×，也不宣称分页稳定加速 1.7%。共同预热、
单进程交替与原独立运行是不同测量设计，新数据只说明旧退化没有稳定复现。

## 单独改变物理页地址

统一分配器、同一个八请求 cohort、相同 teacher-forced token 流，只改变当前页号。
先做 NaN-poisoned cache 和完整 logits 验收，再计时；有效 KV 有限、未拥有页仍
为 NaN、全部引用归还。每进程各布局三次，循环平衡首/中/末位置。

| 当前页布局 | Prefill stream ms | Decode stream 均值 ms | Decode 同步主机均值 ms |
|---|---:|---:|---:|
| stride4：0,4,…,28 | 87.90 | 5.2281 | 5.3524 |
| dense0：0,…,7 | 95.43 | 5.2757 | 5.4330 |
| dense8：8,…,15 | 86.92 | 5.2855 | 5.4123 |

先取每进程三次探针统计量的中位数，再取三个进程中位数。CUDA event 包住的是
`run_model`，含模型、Graph buffer copy 和 LM head，不含 sampler；eager Prefill
的 stream 时间也可能包含主机 dispatch 空档。它不是纯 attention-kernel 时间，
更不能换算成正式服务吞吐。

dense0/dense8 的 Decode 中位差约为 +0.9% / +1.1%，单进程相对差异还会变号；
本轮不支持固定 6.9% 的 Decode 布局惩罚。Prefill 的 dense0 中位数仍比 stride4
高约 8.6%，因此也不能宣称页地址完全没有影响；不能把这些不同边界的探针直接
拼成原完整请求的吞吐预测。

## 能确认与不能确认的边界

可以确认的是：覆盖输入中数值和地址隔离检查通过；没有新增编译；在当前
共同预热/平衡配对条件下，不存在稳定复现的整体 −6.9% 现象。

尚不能唯一归因于主机调度、设备状态、初始化/预热状态或地址局部性。区组外
遥测为 P2、SM 2707～2835 MHz、50～64℃，不是 trace 内连续监测，也没有原始
18-process 时段的对应遥测。未锁频、未改功率策略，不以这些数据反推历史降频。

因此本轮不修改分配器，不删除旧负结果，不将诊断 probe 混入性能排名。H3 的
架构/权重映射和正确性接入可独立继续；未知输出长度、抢占/换出、在线 SLO 与
多卡仍未覆盖。

## 文件与验证

- [summary.json](summary.json)：区组配对比与布局统计；[audit.json](audit.json)：
  完整向量重比、原始指标重算与哈希绑定。
- [raw/run1.json](raw/run1.json)、[raw/run2.json](raw/run2.json)、
  [raw/run3.json](raw/run3.json)：完整 trace、事件、数值元数据、遥测与编译计数。
- 大型张量：忽略的 `runs/validation/h2_kv_diagnostic/h2_kv_diagnostic/`；
  日志：`runs/logs/h_nanovllm/h2_kv_diagnostic/`。
- 旧 18+36+18+18 共 90 份正式 raw 哈希不变，教学大纲无 diff，GPU 已释放。
- 新增 10 项单测通过；CPU 全回归 162 项，160 通过、2 项 CUDA/Triton 跳过。

只读复验：

```bash
runs/envs/nano-vllm/bin/python -m serving.h2_kv_diagnostic --check-only
```

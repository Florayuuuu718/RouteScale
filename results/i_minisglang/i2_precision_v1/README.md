# I2 新精度七配置与本机最终验收

当前状态：**数值资格验证通过，正式性能验收尚未开始，I2 本机尚未关闭。**
HTTP 第三轮的全部功能检查通过，但外部 Windows Python 占用 GPU，独立进程
日志门槛失败。等待 GPU 独占后在新目录重验 HTTP，再运行完整性能矩阵。
本轮未进入 I3，未使用付费资源。

协议见 [冻结数值与计时协议](../../../docs/i2_precision_protocol.md)，重试入口与
服务关闭策略见 [补充说明](../../../docs/i2_precision_retries.md)。原 BF16 对
HF FP32 的失败及此前性能结果均保持原范围，不重写为通过。

## 已验收的数值结果

[qual001/audit.json](qual001/audit.json) 独立重新打开全词表张量，复算全部
25088 个目标向量及原始输入、模型、源码、Graph 捕获、实际执行与资源检查。
16 个独立进程正常结束：两种新参考与两种模式各七配置。

| 模式 | 参考与结果 | 最坏相对 RMSE |
| --- | --- | ---: |
| mixed | 七配置共 12544/12544 向量通过原 HF FP32 门槛 | 0.003614295（0.3614%） |
| bf16_control | 七配置共 12544/12544 向量与新原生 BF16 参考逐元素 exact | 0 |

原阈值保持 `atol=0.125, rtol=0.02, relative_RMSE<=0.005`，并要求全部有限。
控制输入为 prompt512、batch1/3；额外资格输入为 seed20261101、prompt768、
batch1/2。共 28 请求，每个 64 步的全词表 logit。新 HF 控制输入的 1024 个
向量复现旧 HF FP32 参考；新 BF16 参考复现此前 1792 向量的共同形状范围。

混合精度每配置有 1784/1792 个向量的 top1 与 HF 相同。以上证明的是固定
continuation replay 下的数值阈值，不保证自然生成序列相同。BF16 成本对照
只通过内部策略一致性，仍未通过外部 HF FP32 门槛。

## 策略与测量边界

两个模式共同使用 FlashInfer FA2、prefill no-split 和 prefill linear tile512，
包括尾部填充的实际成本。mixed 使用 FP16 线性/QK norm/RoPE/attention/KV/head，
残差与层间/最终 RMSNorm 用 FP32；57 个 norm 在 GraphRunner 构造和捕获之前
安装，独立检查真实 captured graph keys。BF16 control 不安装这些 norm。

混合策略的 norm 使用 PyTorch 多 kernel 实现。其类型转换、发射和计算成本均
应计入正式计时，所得差异包含实现成本，不能概括为 FP32 数学不可避免的成本。
vendor 树仍为 `8ba75c4d14eb6cc985a254fd5185606d62da9061`，策略显式位于 adapter。

七配置为 Naive、Radix、Chunk256、Chunk128、Graph[1,2,4]、Graph[1,3] 和
Overlap+Graph[1,2,4]。正式矩阵预定两模式 × 七配置 × 三独立进程，共 42 次；
顺序采用正序、逆序和轮转，配对精度顺序也平衡。每进程四类固定 cohort 的
cold/warm 各 3 次预热、5 次测量，每请求固定 64 步。

吞吐边界为 scheduler cohort 提交至 host 终态且 GPU 完成，包含调度、native
argmax 和强制 continuation 控制器；不包含 HTTP、tokenizer、启动或 JIT。
Scheduler first-result 不能称为 HTTP TTFT，固定 cohort 吞吐不能替代在线
SLO/goodput。只有三进程样本时须报告范围与 CV，不据小收益宣称通用最优。

## HTTP 重试与检查

- [http001](http001/validation_issue.json)：健康退出超过原 4 秒 SIGINT 宽限，发生升级，失败保留。
- [http002](http002/validation_issue.json)：正常退出，但缺少被覆盖的 `workers_stopped` 观察事件，失败保留。
- [http003](http003/audit.json)：168 请求、560 项功能检查通过，13 个进程结束时有外部 92 MiB 占用，进程日志门槛失败。

关闭修复保持外部 10 秒门槛；内部所有 worker 共享 6+1.5+0.5=8 秒预算。
两项 CPU 关闭测试通过；[真实双进程故障探针](shutdown_fault_probe.json) 在
7.505 秒内清理两个忽略 INT/TERM 的 CPU 子进程，无残留。该故障预算检查与
健康 GPU HTTP 关闭验收分开记录。

CPU 证据：[精度/原始准入 9 项](cpu_tests.json)、[关闭 2 项](shutdown_cpu_tests.json)、
[实际重试准入 9 项](followup_cpu_tests.json)，共 20 项通过。新编排在启动前和
每个子进程结束后执行 GPU 空闲检查，外部占用不会被当成通过。

## 剩余工作与证据位置

等待独占 GPU 后执行 http004；其数值、HTTP、shutdown 三门槛通过后执行
perf001 的 42 个进程。最终独立性能审计还需复算全部数值/HTTP 证据、实际
prefill/cache/Graph replay、每波资源归还及测量标记。预期导出 112 格统计、
56 项配对精度成本、96 项系统比较和三类图，最后生成 acceptance.json。

最终 HTTP 位置通过 provenance 的 `artifact_bindings` 选择；冻结配置中的
http001 仍作为首次默认位置保留，不能覆盖或改名。完整命令见重试补充说明。
源码快照、模型/输入/张量/日志哈希分层绑定；大张量和原始日志放在 ignored
runtime 路径，紧凑结果与审计入 Git。

FA4/TRTLLM 在此前锁定环境的有界 API 探针中不可用，不做 fallback 排名。
TP2/4 留待完成全部本机工作后集中租卡，不由 TP1 结果替代。
复盘、面试指南和个人验收材料仍留待全部实验完成后统一整理。

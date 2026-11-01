# I3 v1：新输入数值失败与精度诊断

本目录不包含正式在线矩阵。原 I2 通过有限输入，并不保证新文本都满足同一门槛；
I3 在新工作负载上重新验收，失败和源码快照均保留，不改阈值。

| 尝试 | 保留结果 |
| --- | --- |
| pilot001 | 两个容量 pilot 执行成功、worker 均 exit0、无升级、GPU归零。独立审计把主动 SIGTERM 后 Uvicorn 父进程的 -15 误当异常，failed audit 原样保留。 |
| qual001 | 3328 全词表向量中3个首步输出超出 relative_RMSE .005，不能启动正式性能。 |
| precision_probe_upper-1 | 首次诊断遗漏隔离环境 PATH，缺 ninja，无数值结果；后续恢复既有环境配置，没有升级依赖。 |
| baseline002 | 12个独立 cold/warm 小探针均通过，max .0042777，但未复现原完整共享KV历史，不能代替失败复现。 |

数值失败：Radix 的 load_023 step0 相对 RMSE 为 .0055648969，load_015 step0
为 .0054680523；Graph 的 load_015 step0 为 .0054680523。逐元素阈值通过，
仍不满足原全部条件。atol=.125、rtol=.02、relative_RMSE<=.005 均不放宽。

随后按原 HTTP 的全部975个实际批次、UID/row顺序、cached/extend几何及原生生成
continuation重放，原Radix1664向量全部逐元素exact，复现同样的两个越界。诊断的
continuation由公共控制器替换，只用于固定原语义历史；不是正式在线生成或性能。

| 诊断策略 | 向量 | 失败 | 最坏相对 RMSE |
| --- | ---: | ---: | ---: |
| 原 I2 mixed | 1664 | 2 | .0055648969 |
| 仅 final norm/head FP32 | 1664 | 2 | .0056817429 |
| 最后8层及 head FP32，FI/KV FP16 | 1664 | 0 | .0049667289 |
| 最后16层及 head FP32，FI/KV FP16 | 1664 | 0 | .0042069010 |

[独立诊断包审计](precision_bundle_audit.json)重新打开原始张量、HF参考和源快照，
确认原失败精确复现、head-only失败保留和候选结果。它的 integrity_passed 仅证明
诊断证据完整，performance_allowed 始终为 false；不将全部误差唯一归因于某个算子。

上层FP32策略选择16层以留出余量，另立 [v2 协议](../../../docs/i3_online_protocol_v2.md)
和 [v2 结果目录](../i3_online_v2/)，增加未参与诊断的输入、重新pilot、重新资格
验证，均通过之后才准许正式18进程。原I2源码与结果不改。

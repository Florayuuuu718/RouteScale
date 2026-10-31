# I1：单卡生命周期修复与验收

2026-10-31，本机 RTX 5060 Laptop 8 GiB，Qwen3-0.6B BF16、TP1，
共享 tokenizer/detokenizer、Graph/Overlap 关闭。最终阶段状态与四场景入口见
[`acceptance.json`](acceptance.json)，补丁和复现说明见
[运行手册](../../../docs/i1_lifecycle_repairs.md)。没有租卡，没有在线性能结论。

原始上游 [attempt004](../i1_lifecycle/attempt004/audit.json) 保留六项失败，
本轮未改写 attempt001–004 或旧配置。修复采用七个可重放提交，
[series.json](../../../patches/minisgl-i1/series.json) 固定补丁及源码树哈希；
干净重放已验证。运行入口独立于原 editable 安装和训练环境。

最终矩阵以 `accept006_healthy`、`accept010_scheduler_fault`、
`accept011_tokenizer_fault`、`accept012_shutdown_active` 为准。故障场景先观测到一个请求
decode，再确认另一个请求实际进入 Scheduler pending，随后注入 SIGTERM。
每个场景均为独立服务进程，串行持有 GPU 全局锁。

健康场景的 33 个请求有 17 finished、4 预期 cancelled、12 rejected。
12 次同时到达的过载请求接受 4 个，其余返回 429；随后正常请求恢复。
非法类型和零输出预算返回 422，输入超窗口或等于窗口返回 400，窗口剩余一位时仅生成一 token，
报告 length。流式和非流式均检查 stop/length；外部 ID 重复提交仍创建新 UID，不提供幂等。

四种取消分别覆盖排队、decode 断开、客户端 deadline 和非流式连接关闭。
审计检查 Scheduler 收到 abort 后不再调度/产出该 UID，table/KV 活跃引用释放，
detokenizer 收到同一有序通道的结束确认后清理，再归还接入名额。
正常完成也要等待前端消费完或关闭，防止慢客户端绕过总上限。
CPU 还覆盖响应头发送失败、body 尚未启动的情况。

总计 40 项本机测试通过：21 项上游控制面/runner、14 项 HTTP/SSE 客户端、
5 项真实证据重算及篡改负对照。见 [CPU 记录](cpu_tests.json)。篡改额外客户端终态、
接入容量、冻结源码和内部关闭升级均被检出。另保留两个失败到通过的修复负对照：
[慢消费者名额](admission_negative_control.json)、[响应头提前断开](headers_negative_control.json)。

## 保留的尝试

| 尝试 | 结果与用途 |
|---|---|
| accept001 | 请求矩阵完成，但 Supervisor 的 2 秒 SIGINT 等待触发 SIGTERM 升级；审计失败 |
| accept002–005 | 第一版四场景通过，汇总留在 acceptance_initial.json；后续 CPU 审查发现 body 未启动边界，故不作最终补丁验收 |
| accept006 | 包含 HTTP 响应所有权修复的最终健康场景 |
| accept007 | 服务故障矩阵已运行，runner 的 psutil PID 扫描竞争导致记录未完成；恢复为明确失败记录，不算验收 |
| accept008–009 | HTTP 同时提交顺序变化，原定 queued 请求先完成；审计检出场景不成立，未计通过 |
| accept010–012 | runner 确认实际 running/pending 后的最终故障及在途关闭场景 |

`acceptance_initial.json`、`cpu_tests_initial.json` 保留当时版本；最终结果只读取
`acceptance.json` 指定的四次运行。被 SIGTERM 终止的 worker 是明确故障注入，
其余业务 worker 必须退出码 0；Supervisor 不能升级关闭信号，runner 不能强杀。
后续故障运行还记录 `/dev/shm/sem.*` 前后集合，核对 resource tracker 清理后的 IPC 状态。

每次运行保存请求、真实 token、资源状态、frontend 快照、执行脚本与补丁快照、上游文件哈希、
依赖/模型哈希和进程退出。逐步观测会扰动时间；finite logits 不证明参考数值等价。
后端/前端清理预算 5 秒，关闭 10 秒，worker 故障客户端传播 3 秒。
在途关闭允许中断响应，客户端应为 failed；不要求强行完成原生成。
可驱逐 Radix 前缀和 CUDA reserved 不要求每请求后清零。

重算单次审计使用该次冻结的 `audit_source.py --directory <目录> --check-only`。
完整 raw JSONL/服务日志位于 `run.json` 的本机 `runtime_dir`；跨机器重算需携带它们。
缺 raw 日志时，35 项纯 CPU 控制面/客户端测试仍可运行，5 项真实证据测试会明确 skip。
最终汇总重算命令见运行手册。

下一步为 I2 本机缓存/调度单变量消融，再做 I3 负载和 H3 计时复验。
Overlap、独立 tokenizer 及 TP2/4 尚未验收；不得只删启动 guard 就宣称支持。
I1 的完成不等于整个 I 阶段完成。个人复盘、面试指南和知识验收按执行计划后置。

## 最终验收汇总

| 场景 | 独立检查 | 客户端终态 | API 关闭 / s |
|---|---:|---|---:|
| healthy | 49 | finished=17, cancelled=4, rejected=12 | 3.087 |
| scheduler_fault | 28 | finished=1, failed=3 | 0.316 |
| tokenizer_fault | 28 | finished=1, failed=3 | 0.219 |
| shutdown_active | 25 | finished=1, failed=2 | 4.017 |

故障场景的 API 关闭列从最终关闭信号起计；故障后的存活 worker 已在 monitor 中提前停止，
其清理时长另外记录在 acceptance.json 的 worker_stop_s。所有场景显存回到 0 MiB，
无 runner 强杀、无 Supervisor 升级，未注入故障的业务 worker 退出码均为 0。

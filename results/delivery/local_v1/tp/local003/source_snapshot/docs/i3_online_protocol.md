# I3 本机 FI/TP1 在线负载协议 v1

用户要求完成 I3，沿用已验收的 I2 混合精度计算策略：FP16 主路径、FP32 残差及
层间/最终 norm，共同 prefill no-split 与 linear tile512。比较 Radix 基线和唯一
变化的 Graph[1,2,4]，共享 tokenizer、page1、2048 页、maxrunning4、context1024、
admission8。Overlap 关闭，供应商树不改，不使用付费资源；TP2/4 仍后置。

## 先冻结工作负载与目标

seed20261101，48 个唯一文本 payload：24 短 prompt/output16，12 长 prompt/output32，
12 共享前缀 prompt/output64；实际 token IDs、长度和 payload SHA 随 workload 归档。
同一原生 Qwen 模型采用 greedy、ignore_eos、分类型固定输出上限；不替换模型生成的
token。四个独立 warmup payload 不纳入正式指标，预热结束和资源归还后才开始正式到达。

在 pilot 和正式结果产生前冻结延迟目标：从计划到达至首个非空 SSE 文本 <=1s，
E2E <=8s，客户端摊销 TPOT <=100ms。三者同时满足且成功结束才计 goodput。
这里的首文本时间明示为文本 TTFT，绝不把 SSE 帧数当 token 数；后端逐 token 发射
时间与 SSE 文本块间隔分开报告。原生 SSE 不提供可靠 usage，实际输入/输出 token
由 tokenized 和 scheduler_token 事件核对，不从字符串重分词猜测。

## Pilot、数值与生命周期门槛

每个策略先用一个新服务进程、并发4的闭环48请求做容量 pilot，仅用于粗定流量，
不当正式开放负载结果或稳态容量结论。取低速率=慢策略 pilot capacity×.25，
near=慢策略×.9，overload=快策略×1.5；两策略使用相同三档速率。near 围绕慢策略，
不能说同时接近两策略饱和。保存原始容量时间段、冻结后的 rates 与 schedule SHA。

在进入18进程正式矩阵前，两策略各起一个资格服务：四个 warmup、全部48种混合长度
请求，分组1/2/3/4，记录实际 batch、Graph capture/replay、KV 命中和资源归还。
只在资格服务安装只读 logits.clone 观察，在请求释放时落原始全词表张量；正式服务
不复制或检查 logits。保留原生 sampler。各策略另起独立 HF FP32 eager 进程，从同一
BF16 checkpoint 升精度，沿该策略实际生成 token 的 continuation 比较每一步完整
151936 词表向量。原门槛 atol=.125、rtol=.02、relative_RMSE<=.005 不改，全部有限。
独立审计重开张量、输入、实际 continuation 和逐步错误，记录 top1，不能声称自然
生成序列与 HF 一致。四个 warmup 加48请求，每策略应1664向量，两策略共3328。

同一资格服务覆盖四个阻塞请求、queued/decode/deadline取消及后续probe，检查客户端
终态、后端停止、请求table/KV、frontend/detokenizer引用、正常退出与无残留。数值
及生命周期独立门槛均通过才准许正式计时，不套用 I2 对新输入的验收。

## 18 次开放到达实验

三档×两策略×三独立服务进程=18次。各 rate 使用同一个无量纲指数间隔 trace 缩放，
所有轮次及两策略复用相同 payload 顺序与计划到达时间；正/逆/轮转 rate 顺序与
交替策略顺序平衡。每进程重启、相同 warmup；GPU 独占，计时期间不并行CPU测试/审计。

客户端一次创建所有请求任务，并发和连接能力不小于48；不因前一请求完成才释放
下一请求。到达后直接发送，客户端预算15s含计划到达后的全部等待，server20s。
原始日志保留 planned/released/http_send/headers/SSE/终态时间及拒绝、错误与取消。
发送滞后 p95<=50ms 且 max<=250ms，否则保留失败轮、查因后新目录重跑，不改变目标。

每个 offered 请求恰好一个客户端终态，不丢失败分母。报告完成 request throughput、
接纳 input token throughput、完成 output token throughput、含部分废弃输出的后端
生成量、文本TTFT/E2E/摊销TPOT p50/p95/p99、scheduler队列时间、后端发射ITL及SSE块
间隔。缓存用实际首次 prefill cached_len，实际 query工作量另计。记录前缀驱逐；
当前 scheduler 不支持 running-request preemption，重prefill/重算观测不可虚构为
抢占机制。区分成功请求的条件分位数与所有 offered 的 SLO 成功率。

吞吐和 goodput 的分母为 max(48/rate, 最后终态-到达原点)，包括 drain；SLO goodput
分子仅为三目标全满足的成功请求。全部48请求（含拒绝/失败/超时）计入SLO成功率
分母。有限到达表和三独立进程只支持本机有限trace结论，p99为经验分位数，不能外推
长期稳态、生产尾延迟或跨硬件最优。必须核对低/near/overload的实际排队与拒绝表现。

## 验收与复现

新目录执行 prepare、pilot、qualification；数值和生命周期独立审计通过，才执行
formal。每阶段保存 source snapshot、模型/输入/配置/日志/张量哈希及进程账本。
任何失败保留原目录及原因，不覆盖、不放宽门槛；实际 gate 目录以显式 bindings 为准。
最终独立审计重算18个服务的每个请求和所有指标，导出完整逐请求CSV、每次运行CSV、
三进程中位/范围、图表、延迟与goodput结果；检查关闭10s、无强杀、无进程/GPU残留。
文档、执行计划及Git推送完成后，标记本机 I3 完成。复盘/面试资料留待全部实验后整理。

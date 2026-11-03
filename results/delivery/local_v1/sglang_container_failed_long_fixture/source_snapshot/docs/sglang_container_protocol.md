# SGLang 容器排队握手验收范围

2026-11-03，`container_queue_handshake_v1`。原生v4的三个通过结果与固定窗口失败保留，
不追认。模型、BF16、FI、TP1、上下文1024、KV4096、运行4/等待8、Graph关闭及上游
源码不改；[原协议](sglang_delivery_protocol.md)的30秒正式客户端、15秒资源释放和
20秒关闭标准继续执行。原48请求的prompt/token IDs/输出预算/near到达表完全保留。

实测容器中，固定0.7秒窗口多次读取同一个负载快照，客户端请求实际已经送达；显式
预热24请求通过后仍复现，因此不能把全部失败归因于JIT。新的诊断先等待4个900-token
blocker实际running（10秒观察预算，原prompt102+900<=1024），再提交明确RID的排队
请求。在5秒内观测到running4且queue>0后，才通过HTTP流任务取消关闭该连接，留下
`observed_queue_connection_close`终态。之后显式原生abort，在另4个请求仍running时
核对queue归零；自动断开与管理中止不混同。900token维持观察窗口，不作吞吐结论。

两种代表prompt、并发4、输出32/64/256以及短prompt并发4输出900，共28个内部预热
请求；预热与模型启动共同使用600秒预算和15秒控制观察开销，正式请求仍30秒。
每个请求的payload、原始SSE、事件和终态落盘；临时服务关闭须无强制清理且GPU0。
准备完成后才用同一声明缓存启动新的完整验收服务。

新[checker](../scripts/check_framework_container.py)与
[独立审计](../scripts/audit_framework_container.py)单独保存。审计保留原生v4所有队列、
原始帧、资源池、flush、进程和GPU门槛，再检查新握手范围与900token fixture。
原v4 checker/auditor及其源码快照不改。本轮不作HF FP32等价或性能排名，不租卡。

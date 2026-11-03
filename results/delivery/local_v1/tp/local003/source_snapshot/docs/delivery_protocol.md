# Mini-SGLang本机交付加固与TP准备验收

本协议对应教学服务的交付加固。规划M5要求的正式SGLang/vLLM框架迁移另行验收，
不能以这里的Mini-SGLang结果代替。

2026-11-03，真实GPU检查前冻结本轮范围。选用I3 v2验收的Mini-SGLang链路，保持
原计算源码、patch tree、FP16主路径/FP32上16层及head、FI/KV FP16、TF32关闭，
Graph[1,2,4]、共享tokenizer、maxrunning4、admission8和1024窗口。
不改旧实验文件或把新交付当成I3性能复测。

交付需包含独立入口、99个锁定依赖、原模型/源码哈希、九补丁重放、readiness、
有期限的启动/停服、HTTP监控、CPU CI及运行手册。仅监听本机127.0.0.1。
`/live`表示API可响应；`/ready`与`/health`保留真实worker存活判断。
监督进程持独占GPU锁，启动验证两个端口、env/model/source/tree；停服有10秒预算，
异常时仅清理自己创建的独立进程组，不停止无关GPU任务，不自动重启掩盖失败。
停止命令必须核对Linux PID启动标识。

新HTTP指标只统计传输状态，不能把200或流结束当后端成功；保留admitted/aborting/
frontend缓冲/worker存活独立gauge。GPU smoke需在新的干净Git副本中重放九补丁，
使用独立新装锁定环境；模型和编译工具链可只读复用并核对哈希。依次检查healthy、
worker_fault和shutdown_active，保存所有请求终态、取消/客户端期限、429、验证错误、
前后端与detokenizer清理、实际Graph/精度、worker退出和GPU归零。失败目录不覆盖。
CPU测试、依赖一致性、真实smoke和独立审计全部通过后才标记本机交付完成。

TP准备是新范围：现有TP1-only guards原样保留。H2/nano新适配器逐rank固定同一
32块逻辑KV池，BF16禁止中间归约降精度、静态页表/paged prefill，并对所有路径
使用共同Decode桶[1,2,4]（Eager的batch3也补4行，head仍只投影真实行），参考为同host TP1
全词表内部一致性；不宣称原BF16/HF FP32外部失败已解决。I2/mini新TP适配器保留
FP16+FP32 upper16/residual/norm，并补齐TP词表gather、每rank一致的prefill tile512、
FI no-split；这是新计算适配器，不能继承旧TP1数值gate。

未来在同一已获授权的多卡host上重跑TP1/2/4：先拓扑与各rank NCCL功能检查，
再冻结相同token fixtures，3个nano策略+7个mini策略共30份资格配置。每份64个
向量，逐元素atol=.125/rtol=.02、relativeRMSE<=.005；mini每rank还需通过独立HF
FP32同continuation参考。nano始终与同host TP1比较，不混排不同精度框架绝对吞吐。
完整独立门槛覆盖全部world/variant、源码、模型和张量哈希后，才允许运行90份
独立性能进程（3轮，预热3/测量5，保留cold/warm）。通信/计算Profiler另开trace阶段，
不混入正式计时。记录per-rank显存、NCCL、拓扑、Graph实际路径和退出状态。

本机仅可验TP1、CPU合同和设备不足拒绝；不能签发完整多卡gate。任何TP2/4缺失、
数值失败或陈旧证据均禁止benchmark/trace，子进程900秒上限并有10秒清理。
本轮不租用GPU、不写个人复盘/面试指南；多卡资源预算/停止规则须在租卡窗口前
另行确定，准备脚本本身不授予付费许可。

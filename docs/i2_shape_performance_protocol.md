# I2 显式形状控制范围的性能协议 v1

本协议在新范围正式计时前冻结。原 HF FP32 全模型门槛仍失败，原 v1 性能进程为0；
这里比较已经独立验证的同一 BF16 引擎内部系统策略，不能声称跨 HF FP32 的误差通过，
也不能称作未改变 prefill 计算形状的上游原生性能。

前置条件是新输入/全部七配置的完整数值资格验证和同一 policy 的全部七 HTTP 生命周期
复验。Controller 核对输入子集、模型、原生树、共同计算参数和 adapter 源码哈希，
失败停止。所有配置显式应用 prefill no-split 与每层四个 linear 的 tile512。

使用资格验证中的 control512 原 token ID 子集：batch1/3、prompt512、输出64、
prefix0/384、cold/warm。七变体、三独立进程，共21进程；每进程3预热、5计入统计的
完整四case cold/warm配对。首轮正序、次轮逆序、第三轮轮转，GPU全局锁串行运行。

计时仍从 cohort 入 Scheduler 到所有 host 终态及 GPU 完成，包含控制器固定续写、
metadata、完整模型、原生 sampler 和 host 结果处理。没有逐步 logits CPU 复制或
finite `.item()`。batch 轨迹在计时后落盘；请求表/页索引、哈希和遥测检查在计时外。
首结果称 Scheduler first-result，不称 HTTP TTFT/在线 SLO。

除实际 prefix 命中、真实 query 数、Graph capture/replay/padding 外，另报告每波
prefill linear 的真实行数和补齐后的行数；tile512 的额外运算计入 wall 时间，不能隐藏。
允许可驱逐缓存与 CUDA reserved 保留；每波所有活跃引用释放且页/table索引守恒。

报告三个独立进程的均值、范围、CV和配对比值范围；进程内五次重复不当作独立样本。
记录温度、时钟、功耗/P-state，出现噪声或交叉排序就报告不确定性，不挑最佳值。
本范围不会替代后续 I3 在线到达负载，也不代表 TP2/4 完成。所有失败和旧数值策略保留。

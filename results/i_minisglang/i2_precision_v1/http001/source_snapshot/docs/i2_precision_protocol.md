# I2 本机混合精度最终验收 v1

用户要求在进入 I3 之前完成 I2 本机范围。本协议冻结 FI fa2、TP1、Qwen3-0.6B、
共享 tokenizer、page1/2048页、maxseq1024/maxrunning4；TP2/4仍留后期集中租卡。
现有 BF16 外部 HF 失败及全部历史证据保留。不开付费资源。

## 计算策略和参考

所有七配置共享 prefill no-split、linear tile512，真实填充成本计入计时。
`mixed` 使用 FP16 主线性/QK norm/RoPE/FlashInfer attention/KV/head，FP32 残差和
层间/最终 RMSNorm。Norm 在 GraphRunner 构造及捕获之前安装，实际 captured GPU
运算与 eager 相同；旧诊断器拒绝已捕获 Graph 的 guard 保留。策略不藏在观察器或
sampler 中，不改变供应商源码。`bf16_control` 保持既有 BF16 shape-controlled 策略，
作为同期成本对照；它仍不是通过外部 HF FP32 门槛的方案。

独立新 HF eager 参考从同一 BF16 checkpoint 升 FP32，覆盖原512-token、batch1/3、
zero/shared、cold/warm，以及新精度未参与定位的768-token、batch1/2、seed20261101。
沿用原 `atol=.125, rtol=.02, relative_RMSE<=.005`，每步、全部151936词表元素有限且通过。
固定64步 continuation，原生 argmax 执行但输出由公共输入控制器替换；不证明自然生成
序列或质量一致。另起独立 BF16 Naive 参考，BF16 控制组必须在七配置下复现其数值。
28请求、1792参考向量；两策略各七配置，共25088个目标向量。记录top1但不替代门槛。
TF32、FP16/BF16 reduced-precision reduction关闭。权重格式变化源自同一BF16 checkpoint。

先做小型 Graph/Overlap smoke，再在新目录冻结源码执行完整资格验证。全部数值完成并
独立重开张量审计后，才复验 HTTP。两策略各七真实服务配置，预期168请求、560项场景检查；
实际 engine precision/policy/source SHA 与 Graph捕获前安装事件必须可核对。覆盖正常EOS、
length、混合请求、decode/queued/deadline取消、后续probe和健康关闭；不声称重跑全部I1故障组合。
客户端终态、后端停止、KV/请求table活跃引用、frontend/detokenizer引用及子进程退出同时检查。

## 正式计时与成本

数值和HTTP独立门槛均通过后，执行两策略×七配置×三独立进程，共42进程。
原512输入子集；每进程3次warmup、5次测量，所有四case及cold/warm。变体顺序依次为
正序、逆序、旋转；策略相邻执行，第二轮反转策略顺序，第三轮交替顺序，减少时间漂移。
持有全局GPU锁；正式计时期间不并行运行测试、审计或其他GPU工作。每波资源审计和
温度/时钟/功率快照在计时之后，快照不等于kernel执行期间时钟。全部策略开销计入实际wall。

报告每个进程均值、三进程中位与范围、CV及全部原始小型结果。相同变体的mixed/BF16配对
吞吐比是迁移到新精度策略的成本对照；BF16的外部数值失败明确保留。每个精度内部再比较
Naive/Radix、chunk256/128、Graph[124]/[13]、Overlap。完整导出112个统计格、全部比较，
不挑胜者。仅3进程时不宣称精细排序或通用最优。Scheduler first-result不叫HTTP TTFT，
固定cohort吞吐不叫在线goodput或SLO。

## 结束条件

七mixed配置全部通过外部HF原阈值；BF16同期控制数值一致；HTTP与退出独立审计通过；
42/42正式进程及资源/实际work/capture/replay/精度策略独立审计通过；完整图表/CSV/复现
命令和结论归档，GPU专用工作结束。此时标记I2本机FI/TP1完成，可开始I3。
原BF16 HF失败、未支持后端、TP2/4延期和新输入仍需验收的边界继续明示。
失败/中断轮用新目录重试，保留源码快照和进程账本。复盘和面试材料待实验全部结束统一写。

# I3 在线协议 v2：新文本数值门槛与上层 FP32

v1 的 pilot、数值张量和失败审计保存在 `results/i_minisglang/i3_online_v1`，
不得覆盖。原 I2 所有冻结记录、源码和配置不改。本版本是新的精度与结果范围，
不是把 I2 的有限输入验收外推为任意文本都通过。

v1 独立重开3328向量发现：Radix 有2个首步输出超过原 relative_RMSE .005，
最坏 .0055648969；Graph 有1个首步超标。元素级阈值通过仍不能替代 RMSE 门槛。
逐批重放原975个实际批次及全部前缀/生成历史，1664个Radix向量逐元素exact复现，
包含相同的两个失败。短孤立probe未复现原错误，保留为不足以替代完整历史的对照。
所有诊断 performance_allowed=false，不进入正式性能结果。

## 新计算范围

`config/serving/i3_online.json` 冻结 `precision.upper_fp32_layers=16`：下层12层沿用
I2 的 FP16 主路径与 FP32 残差/norm；最后N层的线性、层间norm、QK norm、RoPE、
MLP activation，以及最终norm/LMhead使用FP32。FP32层的Q/K/V进入 FlashInfer 前
转换FP16，attention输出升回FP32，KV仍为FP16。TF32关闭，不修改权重原始来源。
上层FP32权重由已加载的同源FP16权重升精度，不声称恢复此前转换丢失的位。
相同975批次的完整诊断中，head-only仍有2失败；upper8全部1664通过但max .0049667，
upper16全部通过且max .0042069。选择upper16留出数值余量，仍须通过下面全量新验收。

新策略显式位于 `serving/i3_precision_v2.py`，在原I2的57个norm安装之后、真正
GraphRunner构造/捕获之前安装。记录 `i3_precision_policy`、N、head/attention/KV
实际dtype、capture keys和源码SHA；不是观察器、sampler或供应商源码的隐藏补丁。
Graph捕获包括相同FP32计算。策略的类型转换、FP32线性和PyTorch多kernel开销均
包含在服务测量中，不与旧I2精度绝对性能混排，也不据此推断FP32数学的最小成本。

## 输入、门槛及在线目标

正式48个payload及四个warmup、SLO目标、到达表规则、18进程平衡顺序、计时和
goodput定义沿用 [v1协议](i3_online_protocol.md)，不删掉诊断中失败的load_015/023。
先用新策略重新pilot再冻结速率，不套用旧精度下的容量数字。

资格验证另增加8个seed20261102的请求，含新长度和批次，独立于上述精度诊断，
每个输出32token。它们只用于资格验证，不混入正式48请求的到达表。每策略资格
共60个正常请求、1920个全词表向量，两策略3840向量；全部沿各自实际greedy历史，
由独立HF FP32 eager参考复核，原atol=.125/rtol=.02/relative_RMSE<=.005完全不变。
同时重新验收取消、引用/资源归还、Graph真实replay和进程关闭。

父API进程由控制器明确发送SIGTERM。Uvicorn清理后可能重发该信号，因此父进程
退出码0或-15且记录planned shutdown均可；worker仍必须exit0、无升级/无残留，
外部关闭仍<=10s。v1把父进程-15误判为异常的审计保留，不能改写成新通过。

## 最终实际入口

prepare、pilot、qualification使用 `scripts/run_i3_online.py` 和新的结果目录。
先运行 `audit_i3_qualification.py` 与 `audit_i3_pilot.py`，两者独立通过后，正式
入口为 `scripts/run_i3_formal.py --attempt online001 --qualification <new-q> --pilot <new-pilot>`。
此入口绑定两个正向审计及其SHA、模型/源码/配置/到达表；原控制器的基础formal
分支不作为最终验收入口。冻结后的源或策略变化必须在新目录重新签发门槛。

18次结束后运行 `audit_i3_online.py`，重开全部数值/生命周期、pilot容量和每个
offered请求的原始证据，导出 `export_i3_online.py` 的CSV/图，再运行
`accept_i3_online.py`。报告原失败、数值边界、全部负载档和失败分母，不放宽
SLO、不到达目的之前降低负载、不虚构未观测的客户端token间隔或稳态结论。
仍不租卡，不开始H3或个人复盘/面试资料。

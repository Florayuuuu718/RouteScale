# 本机服务交付与 TP 准备

2026-11-03。本目录保存新的工程交付证据；不覆盖 G/H/I 的计算实验，不将
Mini-SGLang 加固等同规划 M5 的完整框架迁移。正式 SGLang 采用下述独立范围。

本轮独立无Torch的CPU环境运行51项测试通过；额外TP词表gather的3项CPU张量测试通过。
见 [CPU日志](cpu_tests.txt)、[轻量环境](cpu_environment.json) 和
[张量测试](tp_head_tests.txt)。CPU结果不替代真实GPU证据。

完整 SGLang0.5.20使用新原生BF16范围，依赖/源码/模型已固定，三个真实单卡场景已通过。
对应 [协议](../../../docs/sglang_delivery_protocol.md)、
[运行手册](../../../docs/sglang_serving_delivery.md) 和
[取消源码路径](../../../docs/sglang_cancel_sourcewalk.md)。

## 正式 SGLang 原生验收

固定上游release commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，独立Python3.11.16
环境206个精确依赖；Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`。
范围为原生BF16、FlashInfer、TP1、UnifiedRadixCache FULL，Prefill/Decode Graph均关闭，
上下文1024、KV4096、运行4、排队8。此范围是功能交付，不继承I3 upper16的HF FP32资格，
不据此排列两个框架的性能。独立干净checkout的实际验收commit为`14eef4a`。

|场景|规范客户端终态|独立检查|启动 / 关闭秒|结果|
|---|---|---:|---:|---|
|[原生正常服务](sglang_native_healthy/audit.json)|63完成、3取消、5失败|598|66.119 / 4.493|通过，5次均为实际队列拒绝|
|[原生worker故障](sglang_native_worker_fault/audit.json)|1完成、1失败|53|63.114 / 0.668|预期failed，审计通过|
|[原生带请求停服](sglang_native_shutdown_active/audit.json)|2完成|54|61.141 / 11.817|排空后stopped，审计通过|

合计75个规范请求，66完成、3取消、6失败；最后6次为5次队列拒绝和1次受控worker故障。
三个场景均无强制清理，GPU结束0 MiB。另有HTTP非流式与参数验证等诊断，不混入规范
流式账本分母。原I3的48个prompt、token IDs、16/32/64输出预算及near到达表保持不变，
只映射model alias；48个均完成。near速率沿用I3，并未标定成SGLang饱和点。

真实队列诊断先观测4个原生running，再让已知RID入队。该客户端关闭后显式调用原生
`/abort_request`，核对另4个仍running时queue从1归零，保留两种动作，不能据此声称
自动断开处理已在同一时刻完成。独立断开和deadline场景沿原生路径清理。过载拒绝实际
通过HTTP200中的SSE error(code503)传递，客户端记failed；独立拒绝分类读取原始错误，
没有把200当生成成功，也没有把其他503都归为队列满。

每个offer在发送前落盘catalogue，每个事件即时落盘；独立审计重建全部原生SSE文本、
终态和payload绑定，并核对running/queue、原生flush、真实pool事件及KV指标。原生空闲
指标30秒节流曾使已flush的池仍展示旧值；小型观测补丁成功flush后调用原生采样器，
从实际pool读取数值，失败flush不刷新，CPU非零反例防止写死零。上游wheel源码未修改。

[时间线](sglang_native_healthy/native_timeline.png)来自78个实际运行期样本；
[离线仪表盘](sglang_native_healthy/dashboard_snapshot.html)可直接保存后打开，
[导出清单](sglang_native_healthy/export.json)固定三份产物SHA。图中running不超过4、
queue不超过8、KV used峰值1900/4096；运行期间GPU总占用2538–2564 MiB。
停服后的GPU0由status单独证明，不在曲线中虚构采样点。混合诊断流量的条件延迟不是
生产SLO或稳态性能结论。

## 正式框架失败与环境证据

以下失败保留原目录，不追认成成功：

- [Ninja路径](sglang_failed_ninja/run.json)：环境bin未加入PATH，API启动前失败。
- [首次JIT与watchdog](sglang_failed_watchdog/run.json)：初始30秒预算不足；恢复原生300秒
  watchdog，启动总预算600秒。完整无缓存启动另一次实际270.439秒成功，编译成本保留。
- [flush后的旧gauge](sglang_failed_idle_metrics/run.json)：模型已ready且能生成，失败在
  15秒settle读到旧KV指标；按真实pool刷新修复，未降低资源释放标准。
- [启动接口](sglang_failed_entrypoint/run.json)：固定版本没有所猜测的`.main`；改用其
  实际prepare_server_args/run_server/kill_process_tree路径，并真实执行CLI help。
- [未观测到四个运行请求](sglang_failed_running_observation/run.json)：原900token诊断
  不满足触发条件且停服升级，不能验收；新有界256token诊断保存实时loads及持久客户端账本。
- [HTTP状态断言](sglang_failed_HTTP_status_assertion/run.json)：实际是HTTP200/SSE503，
  原断言要求HTTP503而失败；按原始帧识别错误，没有改服务器响应或成功标准。

大型依赖下载曾反复断流。[续传脚本](../../../scripts/fetch_framework_wheels.py)以官方
PyPI分段下载八个大型wheel，再核对完整文件SHA；
[完整校验](software_verified.json)、[来源](software_pypi_wheels.json)和
[离线安装](software_sglang_offline_install.txt)保留。206个锁定包经兼容性检查，
19份原生关键源与固定release逐一对应。依赖目录、编译缓存和模型不入Git。

容器的最终检查收据在构建及实际GPU场景结束后归档；官方stock镜像与离线配方的验证
状态分别记录，不能从Dockerfile存在推断运行通过。
首次镜像导出触及1200秒构建超时；重试后成功获得固定镜像ID。首次容器又被Windows
1971端口占用阻止，改宿主机21971/21972映射，不终止原监听进程。随后实际模型加载
成功，但首次CUDA RoPE编译因较新glibc专用的头文件声明与Bookworm冲突而失败，见
[头文件失败](sglang_container_failed_header/native_server.txt)。容器改用原始CUDA13头，
本机兼容头不改，二者各记SHA；所有失败GPU均归零，没有把未ready当作验收通过。

## Mini-SGLang 交付验收

同 I3 v2 的模型和计算源码，九补丁独立重放 tree
`8ba75c4d14eb6cc985a254fd5185606d62da9061`；upper16 FP32、下12层 FP16、
FI/KV FP16、TF32关、Graph[1,2,4]、运行4/总接入8。独立重装99个锁定依赖，
见 [环境安装](environment_install.json)。实际源码副本为干净 Git checkout，
模型、CUDA13工具链和全局锁明确挂载；依赖环境不含 editable 上游。

最终三个场景运行在源码 commit `8b5f2d3` 的同一干净副本。

|场景|流式客户端终态|独立检查|启动 / 关闭秒|结果|
|---|---|---:|---:|---|
|[正常服务](mini_healthy/audit.json)|11完成、2取消、4拒绝|89|249.126 / 3.283|通过|
|[worker故障](mini_worker_fault/audit.json)|1完成、1失败|21|27.080 / 0.466|预期 failed，审计通过|
|[带请求停服](mini_shutdown_active/audit.json)|1完成、1失败|21|26.577 / 4.132|预期 stopped，审计通过|

共21个规范流式账本请求，另有非流式和两类输入验证共3次 HTTP 请求；内部启动预热
单独记录。覆盖正常流式/非流式、超长/非法参数、真实断开、客户端期限、12并发中的
4次429、有界接入、取消后探针、worker死亡传播及带请求停服。每个客户端恰好一个终态。
独立审计重开原始 scheduler、detokenizer、前端资源、Graph replay 和精度事件，
不仅检查 HTTP 返回。三个场景结束均无强制清理，worker退出0，GPU回到0 MiB；
API父进程预期 SIGTERM 的返回码-15明确保留。

## 保留的失败与修复

- [首次冷启动失败](failed_cold_start/run.json)：Decode Graph完成并不意味着首次
  Prefill JIT完成；原公共20秒请求期限返回504。新入口在公开 ready 前执行真实
  Prefill+Decode，内部启动预算600秒，公共期限仍20秒。
- [独立冷启动证明](cold_start_proof/audit.json)：无现成 JIT 缓存，257.685秒后
  ready，后续请求正常；89项审计通过。实际等待成本保留，不称瞬时启动。
- [端口预检失败](failed_port_preflight/run.json)：上一服务关闭后的 TIME_WAIT
  被误判为活跃监听；新预检设置 SO_REUSEADDR，CPU真实套接字测试同时保证仍拒绝
  活跃占用。最终三个场景可连续启动，无需用户手动清理端口。

HTTP监控统计传输结束与状态，不能把200当生成成功；真实后端/资源状态另有gauge和
事件。默认 localhost1969，显式 PID identity 的停止命令，独立进程组与GPU互斥锁。
使用方式见 [Mini-SGLang手册](../../../docs/serving_delivery.md) 与
[协议](../../../docs/delivery_protocol.md)。

## TP 租卡前准备验收

[本机资格审计](tp/local004/local_audit.json)覆盖10个配置、640个全词表向量与70项
独立检查；12个子进程（fixture、HF参考、10配置）均退出0，无升级信号，结束GPU为0。
Mini7策略共448向量通过同checkpoint独立HF FP32原门槛，最大relativeRMSE
0.00120082；nano3策略共192向量为同host TP1内部一致性，不宣称HF等价。
[TP1集合通信](tp/collective001/logs/collective_tp1.process.json)已实际执行NCCL
all-reduce，记录GPU UUID、NCCL和per-rank显存。TP1是退化集合通信，不证明跨卡链路。

新TP代码不改旧TP1-only guards。上层FP32词表按rank gather后重排并裁去padding，
另有world1/2/4的CPU张量检查。nano新适配器固定共同Decode桶[1,2,4]，包括Eager
batch3补4行，成本纳入将来的计时；mini保留upper16和共同prefill tile512。
[未来矩阵](tp/plan.json)包含同host TP1/2/4的30份资格、90份三轮性能进程，
预热3/测量5，cold/warm分别保留，Profiler另开阶段。只有完整独立资格才能解锁性能。

失败完整保留：[local001](tp/local001/validation_issue.json)是torchrun rendezvous
误用agent store；[local002](tp/local002/validation_issue.json)是nano BF16 Decode
batch3与Graph补4行的几何差异，阈值未放宽；[local003](tp/local003/validation_issue.json)
缺少最后进程终态，未验收，补跑尝试在发现原raw已存在时中止且未覆盖；最终新目录
local004完整重跑。各目录均保留实际source snapshot。全量张量不入Git，原始`.pt`
位于忽略目录，SHA固定在审计中，可按同配置重新生成。

真实[TP2设备不足拒绝](tp/tp_absent_tp2.log)和
[单卡不得解锁计时](tp/tp_local_timing_rejected.log)均已执行。正式多卡资格、性能、
通信trace进程仍为0，未租卡、未授予付费许可。本机准备完成不代表多卡接入已完成。

## 保护范围

[历史清单](historical.json)固定1498份旧 G/H/I JSON的SHA256，本轮结束须全部逐文件
复核。旧失败和数值门槛不追认。TP2/4和付费资源尚未验收；正式 SGLang 使用不同
原生 BF16 范围，不能继承上述 I3 混合精度数值结论。个人复盘和面试材料仍等待实验完成。

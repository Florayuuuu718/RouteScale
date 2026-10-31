# I1 单卡服务基线：复现完成，生命周期验收未通过

2026-10-24，RTX 5060 Laptop 8 GiB / sm_120，Qwen3-0.6B，BF16、TP1。
有效完整基线为 [attempt004](attempt004/run.json)，[独立审计](attempt004/audit.json)
有 18 项通过、6 项失败、2 项 unavailable。`lifecycle_complete=false`，不作在线性能结论。
H2/I2 多卡实验统一留到后期租卡；本轮未使用付费资源。

## 已验证的范围

- 独立环境 Torch 2.9.1+cu129 / Transformers 4.57.3 / FlashInfer 0.7.2.post1 /
  sgl-kernel 0.3.21；训练与 nano-vLLM 环境版本保持原样，见 [环境记录](environment_isolation.json)。
- 固定 Mini-SGLang `9a91cfafe754aa85daee49998176275667eb58f2`，Qwen revision
  `c1899de289a04d12100db370d81485cdf75e47ca`；权重、配置、源码与依赖见 [provenance](attempt004/provenance.json)。
- 实际运行 `FlashInferBackend`，内部 fa2，Radix、page-size=1、2048 页、窗口 1024；
  Graph/Overlap 关闭，running slot=1，客户端并发上限 4。未替换 attention backend。
- 15 个客户端请求：9 finished、5 cancelled、1 rejected，每个恰有一个终态。
  cancelled 包括预期取消/断开，也包括超上下文静默丢弃和 worker 故障后的客户端超时，
  因此这个计数不能解读为异常处理全部通过。
- 463 个真实输出 token / 463 次 batch logits finite 检查通过；服务端 prompt IDs 与
  固定 thinking=False 模板的本地 IDs 完全相符。这里只验证有限输出和生命周期，不验证逐 logits 参考等价。
- EOS 短请求输出 2 token（含 EOS）；输出预算请求恰为 32 token；三个并发短请求各 9 token。
- 排队中取消 UID6 从未入模；decode 断开 UID7 输出 8 token 后停止；deadline UID8 输出
  4 token 后停止。三者在客户端终态后约 0.152 / 0.111 / 0.211 秒到达 Scheduler abort，
  均小于冻结的 5 秒清理预算，abort 后没有新的该请求 batch。
- 故障注入前，table 全部可用、pending/running 为空；1483 free pages + 565 可驱逐
  prefix tokens = 2048，总活跃 prefix 锁为零。可复用 Radix 前缀保留不算泄漏。
- runner 最终回收全部专用进程，GPU 占用回到 0 MiB；关闭需要强制回收，不能算优雅关闭通过。

API 就绪耗时 21.108 秒；首个真实请求含剩余 JIT 编译耗时 134.547 秒，之前失败尝试
已留下部分编译缓存，因此这不是空缓存冷启动基准。加载、编译不计入稳态性能。
本轮逐步日志与 finite 检查会扰动时间，0.1–0.2 秒也仅作本次清理时限证据。

## 六项验收缺口

| 缺口 | 实测证据 | 源码定位 / 下一步修复 |
|---|---|---|
| 前端完成请求残留 | 9 个已完成 UID 仍在 ack_map / event_map | API 外层读到 finished 后 break，`wait_for_ack` yield 后的删除逻辑未必执行；将清理放入 finally 并测试正常/异常退出 |
| detokenizer 取消残留 | 后续请求结束后 decode_map 仍保留 UID7、8 | tokenizer worker 仅转发 AbortBackendMsg，没有清理 detokenizer UID；补取消通知与尾部消息处理 |
| finish_reason 误报 | 32/384 token 预算耗尽均报告 stop | API 写死 stop；由后端传递 EOS/length 原因并核对真实 token 计数 |
| 超上下文静默丢弃 | 1113-token 输入超过 1024，HTTP 已返回 200，最终仅客户端超时 | Scheduler warning 后直接 return；在入队前拒绝或向前端发明确错误终态 |
| worker 故障健康检查失真 | SIGTERM GPU worker 后 `/v1` 仍为 200 / ok；客户端 3 秒超时 | `/v1` 是静态响应；补 worker 存活/readiness 检测、停止接入与在途失败传播 |
| 故障后的关闭超预算 | SIGINT 后 10 秒仍未退出，runner 强制 kill | 定位 ZMQ linger、listener/task 与非 daemon 子进程退出顺序；另补健康状态关闭用例 |

最后一项的源码原因尚未唯一证明；只确认本轮含 worker 故障的关闭没有在预算内完成。
以上不是数值门槛失败，也不回写 H 阶段结果；当前只定位原生版本行为，没有对上游做生命周期修复。

两个 unavailable：上游没有配置接入队列容量，running slot=1 不能证明 pending 有界；
重复 X-Request-ID 被分配为两个新 UID（11、12），不能声称原生提供幂等去重。
本轮不测重启自动恢复、非流式取消、TP2/4 或 Overlap 开启时的取消竞争。

## 过程与证据

| 尝试 | 结果 | 处理 |
|---|---|---|
| [attempt001](attempt001/failure_classification.json) | 首 kernel 找不到 ninja | ninja 已在新环境；启动 PATH 漏加环境 bin，修复 launcher，原记录不改 |
| [attempt002](attempt002/failure_classification.json) | CUDA 13.0 / glibc 2.43 rsqrt 声明冲突 | 本地工具链副本仅补两处 noexcept；系统 CUDA 不变 |
| [attempt003](attempt003/failure_classification.json) | 120 秒首请求预算在 JIT 中耗尽 | 保留失败；后续独立冻结 600 秒编译预算，常规请求预算仍为 30 秒 |
| [attempt004](attempt004/audit.json) | 完整正常/异常矩阵复现 | 6 项验收失败如上；不把复现完成等同于 I1 完成 |

工具链原/新头文件哈希在 [toolchain_patch.json](attempt004/toolchain_patch.json)，
准备脚本与部署环境说明见 [运行手册](../../../docs/minisglang_service_walkthrough.md)。
权重与完整日志仍在 `runs/`；紧凑的请求、事件和哈希在结果目录。
attempt004 冻结的依赖文件包含当时本机 editable 路径；当前
[`i1_requirements.txt`](../../../config/serving/i1_requirements.txt) 已将该行改为仓库相对路径，版本不变。

[requests.json](attempt004/requests.json) 包含客户端事件与输出，
[workload.json](attempt004/workload.json) 包含 prompt、IDs、模板/采样哈希，
[server_events.json](attempt004/server_events.json) 包含实际 UID 与后端状态。
完整本机原始日志路径 `runs/logs/i_minisglang/i1_lifecycle/attempt004/`。
审计重算核对这些事件、原始日志和源码快照；在另一台机器重算时需携带对应本机日志。

客户端 [11 项 CPU 测试](client_tests.json) 通过；审计的
[三项篡改负对照](audit_negative_controls.json) 均检出（源码、终态、服务端事件）。
另在原 nano-vLLM 环境重跑 H3 v5 的 check-only，36 个正式进程的历史审计仍通过。

```bash
runs/envs/mini-sglang/bin/python -m unittest discover -s tests -p test_online_client.py -v
runs/envs/mini-sglang/bin/python scripts/audit_i1_lifecycle.py --directory results/i_minisglang/i1_lifecycle/attempt004 --check-only
```

下一步按表中顺序做独立修复范围，保留本轮基线；补有界接入、错误传播与健康服务关闭。
I1 验收收口后进入 I2 单卡消融，再进入 I3 负载协议；TP2/4 统一后期租卡。

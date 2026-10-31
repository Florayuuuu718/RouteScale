# I1：Mini-SGLang 单卡服务生命周期基线

本轮进入 I，先检查本机单卡真实 HTTP/SSE 链路。配置冻结在
[`i1_minisglang.json`](../config/serving/i1_minisglang.json)。H2 与 I2 的多卡实验统一留到
后期租卡，同机补测 TP=1/2/4；本轮不租卡，不把单卡 smoke 当作整个 I 完成。

上游固定为 `9a91cfafe754aa85daee49998176275667eb58f2`，MIT；复用 H1 固定的
Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`，Apache-2.0。
独立环境 `runs/envs/mini-sglang`，不修改训练和 nano-vLLM 环境。
实际依赖、GPU、模型文件哈希与源文件快照写入每次 attempt。任何安装/启动失败保留日志，
新尝试写新目录。不开正式性能矩阵。

初始配置：BF16、TP1、FlashInfer `fi`（其源码指定 fa2）、2048 个 page-size=1 的页、
上下文 1024、单个 running slot、prefill chunk 256、Graph 关闭、Overlap 关闭、Radix 开启。
这些设置用于明确排队与取消边界；尚未比较任何策略收益。真实 kernel 不支持时保留失败，
不把 fallback 冒充该 backend 的结果。

启动预算 600 s、首请求冷编译预算 120 s、取消/断开清理预算 5 s、关闭预算 10 s、worker 故障客户端预算 3 s；
全局 GPU 锁沿用 `runs/tmp/h2_runner.lock`，一次只运行一套服务。冷启动/编译计时独立记录。
退出后核对进程树与 GPU 内存，必要时由 runner 在预算内回收专用子进程，并明确记录强制回收。

客户端在计划到达时进入 `queued`（客户端 semaphore），第一次 SSE choice 后进入
`running`（仅客户端观测），最终仅一个 `finished/cancelled/failed/rejected`。
请求超时包含 semaphore 等待；总预算从计划到达计。断开/定时取消实际关闭 HTTP 流。
输入 request ID 在发送前去重，通过 X-Request-ID 发送；上游不提供外部 ID 幂等保证，
用实际响应 `cmpl-<uid>` 和逐请求 tokenizer/scheduler 日志关联，不捏造服务端去重能力。
必须收到 finish_reason 与 `[DONE]` 才计完整结束，SSE frame 数不当作 token 数。

使用本地 tokenizer 预先 apply_chat_template(enable_thinking=False)，将得到的原始 prompt
发送至 chat endpoint 的 `prompt` 字段，以免上游默认 thinking 模板改变负载。
记录原始文本、最终 prompt、token IDs、模板及所有采样参数哈希，比较服务端实际 IDs。
贪心输出只用于链路验证，不与 H 的数值/性能结果混为一表。

观测通过 `serving/minisgl_observer.py` 包装固定版本的方法，所有原方法原参数调用，
不改调度/采样/资源释放行为。新增仅本机调试 GET `/debug/i1/frontend` 返回前端 map。
按 PID JSONL 记录 scheduler 收到请求、prepare batch、真实 output token、释放、idle
完整性检查、tokenize IDs 和 detokenizer 活跃 UID。host monotonic 时钟只解释控制面区间，
不冒充 CUDA kernel 时间；没有原生观测的阶段标 `unavailable`。
每步日志会扰动延迟，因此本轮的时间仅用于 deadline/清理判定。

先用 CPU 假流测试分片、多行 SSE、异常结束、HTTP 拒绝、超时、并发上限、去重及关闭连接。
真实服务依次验证短输出 EOS、强制预算结束、并发、排队中取消、decode 断开、超时、
非法类型参数、超上下文请求、重复外部 header、worker 退出和关闭。
负面测试有独立有限超时，不能拖住整个 runner。上游若静默截断/丢弃、漏报终态、
保留活跃引用或关闭不完整，报告失败/限制，不改变通过标准。

验收分别记录 transport、后端计算停止、table/KV 活跃引用、frontend/detokenizer 清理、
worker 故障检测和有界接入。不要求 Radix 可驱逐缓存或 CUDA allocator reserved 归零。
只有这些项均有证据才能把 I1 生命周期标为完成；存在已定位缺口时只能完成“上游基线复现”。
I2/I3 应等待相应生命周期缺口处理后再做正式在线性能结论。

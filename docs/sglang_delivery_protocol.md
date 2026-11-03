# 正式 SGLang 本机交付验收协议

这是规划 M5 的完整框架迁移，与 Mini-SGLang 教学服务交付分别验收。
2026-11-03，在实际模型启动前确定本范围：SGLang 0.5.20，源 commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，独立 Python 环境及锁定依赖；
同一 Qwen3-0.6B checkpoint/tokenizer，revision 与十文件 SHA 见
[模型清单](../config/delivery/model.json)。原 I2/I3 计算和证据不变。

采用原生 BF16、FlashInfer attention、TP1、1024 上下文、运行上限4、等待队列上限8、
KV4096 token、chunked prefill256、静态显存比例0.55。Prefill/Decode CUDA Graph
均关闭，这是功能交付范围，不继承 I3 upper16 数值资格，不与 I3 作吞吐排名。
配置见 [runtime.json](../deploy/local/sglang/runtime.json)。原生 API 与 dashboard
只在本机监听1971/1972，启动预算600秒，客户端预算30秒，关闭预算20秒。
实际健康生成通过后才发布 ready；监督器独占原 GPU 锁，只清理自己的进程组。
所有失败目录保留，调整计算配置必须另开范围和新目录。

使用 I3 v2 的48个原 prompt、采样参数和16/32/64输出预算，复用原 near 到达表。
仅将 model 路径映射为明确的 served alias；原 prompt 哈希与映射前后 payload 哈希
分别保存。继承的1.4800555 req/s只是有限代表 trace，不是 SGLang 饱和率。
通过原生 `/v1/completions` 发送已经套模板的文本，避免二次套用 chat template。
客户端沿用 M3/I3 created/queued/running/finished/cancelled/failed/rejected 事件，
薄适配层保存全部原生 SSE 帧，再映射 text。first_text TTFT 与非空 SSE 文本块间隔
分别报告，不冒称逐 token ITL。全部 offered 请求进入终态分母，包括过载503。

首次v1真实启动在模型加载后因PATH缺少独立环境bin、找不到已安装Ninja而失败，
未进入API流量或数值资格。v2明确将该bin加入PATH，doctor验证Ninja实际可执行，
设置本版本新的SGLANG_CACHE_DIR。另关闭原生可选py-spy/CUDA崩溃dump：其默认诊断
等待60秒会超出20秒关闭预算。worker故障检测、原生stderr和客户端失败记录仍保留。
算法、模型、精度、队列、上下文和客户端期限不改；失败v1单独归档，不追认为通过。
v2随后完成模型/KV初始化，但首次FlashInfer编译在30秒watchdog下中止，保留部分NVCC
产物及失败。v3恢复固定上游默认watchdog300秒，使用新的独立缓存目录；600秒启动、
30秒客户端、20秒关闭预算不变。watchdog约束单次forward失去进展的时间，不能当请求SLO。

验收包含真实非流式、错误预算、超长输入、流式断开、客户端期限、实际队列占用、
过载拒绝、原48请求 trace、取消后恢复、受控 worker 故障和带请求停服。
排队复现先观察4个实际运行请求，再让指定原生RID入队并断开客户端；随后显式调用
原生 `/abort_request`，分别记录断开与管理中止，核对阻塞请求仍运行时该队列项消失。
这不把显式中止误写为原生自动断开的响应时延。
队列清空不等于 KV 释放：在原生 running/queue 均归零后，调用 `/flush_cache`，
再核对原生 KV 使用归零；同时保存缓存可用量、原始 exporter 和完整日志。
正常服务、故障与停服分别保存，不通过强制清理掩盖失败。

监控只使用本版本实际导出的原生名称，缺失保留 missing；框架 processed/aborted
不能直接当客户端成功/失败。原生 TTFT sum/count、queue、running、output、KV 和
模型/KV/Graph 显存分别展示，实际 GPU 总占用另以 nvidia-smi 留样。
取消源码链路需追到 scheduler 和资源释放，结合实际请求与缓存清空证据；
提交一个有 CPU 测试且经真实请求使用的小改动或可重放问题复现。

独立环境与固定 OCI 摘要容器分别记录实际验证状态。容器不得嵌入模型，权重只读挂载，
仅声明自己的 GPU、端口和状态卷。干净 checkout 的 CLI 不依赖隐含个人路径；
外部环境、模型和 CUDA 工具链挂载必须明确。CPU CI 覆盖配置、事件/指标 schema、
汇总 fixture、文档链接和静态编译，GPU 证据单独保存。
本轮不租卡，不将 TP1 通过解释为 TP2/4 验收，不写个人复盘或面试指南。

# 完整 SGLang 交付运行手册

范围见 [正式框架协议](sglang_delivery_protocol.md)，原始工程证据与状态见
[交付报告](../results/delivery/local_v1/README.md)。这里的 SGLang0.5.20 是独立完整框架，
原生 BF16；Mini-SGLang 的 upper16 精度/性能结论不迁移到此处。

## 独立环境与模型

需要 Linux、Python3.11.16、支持 CUDA13 的驱动、单张独占 NVIDIA GPU。
验收硬件为 RTX5060 Laptop8GiB；本配置限制KV4096、运行4、队列8、上下文1024，
不将短 trace 解释为长期容量。环境、模型和编译工具链不在 Git 中。

```bash
uv venv --python 3.11.16 runs/envs/sglang-framework-v1
uv pip sync --prerelease=allow --torch-backend cu130 --python runs/envs/sglang-framework-v1/bin/python deploy/local/sglang/requirements.lock
uv pip check --python runs/envs/sglang-framework-v1/bin/python
```

将 Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`
下载到 `runs/models/Qwen3-0.6B`，与 [十文件清单](../config/delivery/model.json)核对。
CUDA13.0工具链由 [准备脚本](../scripts/prepare_i1_toolchain.py)生成兼容头文件副本，
默认位置 `runs/toolchains/cuda-13.0-i1`，或显式设置 `ROUTESCALE_CUDA_HOME`。
脚本不会改系统 CUDA。首次编译需要可用 C++ 编译器，网络离线时须提前备齐 JIT 依赖。

## 启动与关闭

```bash
python3 scripts/framework_service.py doctor
python3 scripts/framework_service.py run --run-dir runs/sglang/local001
# 另一个终端
curl -f http://127.0.0.1:1971/ready
curl -f http://127.0.0.1:1971/health_generate
curl -f http://127.0.0.1:1971/metrics
python3 scripts/framework_service.py status --run-dir runs/sglang/local001
python3 scripts/framework_service.py stop --run-dir runs/sglang/local001
```

doctor逐文件核对模型、锁定依赖、SGLang版本和11份原生关键源码。run前台持原全局
GPU互斥锁，执行原生 `sglang.launch_server`；`/ready` 加实际 `/health_generate`
通过后才发布ready。端口1971为原生API，1972为本地仪表盘。启动预算600秒，关闭
预算20秒，Ctrl-C也可停止；只清理自己创建的进程组。不要复用旧run目录。
worker异常会使服务失败并退出，排查日志后用新目录启动，不自动重启覆盖故障。

```bash
curl -N http://127.0.0.1:1971/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"RouteScale-Qwen3-0.6B","messages":[{"role":"user","content":"Tell a short story."}],"max_tokens":32,"temperature":0,"stream":true}'
```

验收使用 `/v1/completions` 接收 I3 已套模板的 prompt，不再套 chat template。
普通用户消息可使用上面的原生chat入口。实际兼容性以本版本API和实测范围为准。

## 原生监控与客户端观测

仪表盘位于 `http://127.0.0.1:1972`，原始 exporter 为API端口的 `/metrics`。
实际名称与解析在 [framework_metrics.py](../serving/framework_metrics.py)：

|含义|SGLang0.5.20实际名称|
|---|---|
|运行 / 等待|`sglang:num_running_reqs` / `sglang:num_queue_reqs`|
|processed / aborted|`sglang:num_requests_total` / `sglang:num_aborted_requests_total`|
|输出 token / 速率|`sglang:generation_tokens_total` / `sglang:gen_throughput`|
|TTFT|`sglang:time_to_first_token_seconds_sum` / `_count`|
|KV比例 / 活跃占用 / 可用|`sglang:token_usage` / `sglang:kv_used_tokens` / `sglang:kv_available_tokens`|
|模型 / KV池 / Graph GB|`sglang:weight_memory_usage_gb` / `sglang:kv_cache_memory_usage_gb` / `sglang:graph_memory_usage_gb`|

未导出的指标显示缺失，不补零；processed/aborted不能推算客户端生成成功/失败。
验收显式指定 `ROUTESCALE_CLIENT_LEDGER`，另显示该账本中已经落盘的完成、失败、拒绝、
取消与总offer；它不代表未观测的外部流量。实际GPU总占用由nvidia-smi采样，区别于
框架显存分配项。TTFT原生首token均值与客户端首非空文本分别保存；SSE块间隔不叫ITL。

取消后先等原生 running/queue清空，再调用 `/flush_cache?timeout=3`，核对活跃KV与
使用比例归零。Radix缓存保留可复用前缀不等于泄漏；只有客户端关闭也不足以证明释放。

## 固定镜像与只读模型

[官方配方](../deploy/local/sglang/Dockerfile)固定官方 SGLang 的linux/amd64摘要；
原始索引、平台摘要与离线基础镜像见 [image.lock.json](../deploy/local/sglang/image.lock.json)。
每条构建路径的实际GPU验证状态须查看交付报告，不能从配方存在推断已经通过。

```bash
export ROUTESCALE_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
docker compose -f deploy/local/sglang/compose.json up --build
docker compose -f deploy/local/sglang/compose.json down
```

Compose只声明GPU0和localhost1971/1972，模型bind只读，状态和编译缓存独立卷，
stop grace30秒覆盖监督器20秒关闭。模型不嵌入镜像。

离线配方在已安装且通过doctor的环境中打包精确依赖和工具链。先提交源码保证干净
checkout，再执行；输出目录需有充足空间，归档和镜像均不入Git：

```bash
python3 scripts/prepare_framework_bundle.py --directory /absolute/path/to/new-build-context
docker build -f /absolute/path/to/new-build-context/deploy/local/sglang/Dockerfile.offline \
  -t routescale-sglang:offline-v1 /absolute/path/to/new-build-context
docker run --name routescale-local001 --gpus '"device=0"' --shm-size=1g \
  -p 127.0.0.1:1971:1971 -p 127.0.0.1:1972:1972 \
  --mount type=bind,source="$ROUTESCALE_MODEL_DIR",target=/models/Qwen3-0.6B,readonly \
  routescale-sglang:offline-v1
docker stop --time 30 routescale-local001
```

离线基础为固定摘要Python3.11.16-bookworm；依赖归档和CUDA工具链归档逐一SHA绑定，
镜像内 `/image_manifest.json` 记录干净源码commit和来源。Windows Docker Desktop未
开启WSL集成时，构建目录及权重先复制到Windows可见的任务目录并核对SHA，传入Windows
绝对路径，不需要修改WSL设置。只停止/移除本次自己创建的实例。

## 重放验收与 CPU CI

```bash
runs/envs/sglang-framework-v1/bin/python scripts/check_framework.py --scenario healthy --directory runs/framework/healthy001
runs/envs/sglang-framework-v1/bin/python scripts/audit_framework.py --directory runs/framework/healthy001
# worker_fault / shutdown_active分别用新的目录，GPU串行
```

保存源快照、全部原生SSE、客户端事件/终态、原始metrics、缓存flush、进程状态和GPU归零。
CPU CI不安装Torch、不下载模型；运行tests/delivery与check_delivery_cpu.py，覆盖配置、
事件/指标schema、汇总fixture、文档链接和AST。真实GPU及容器证据单独归档。

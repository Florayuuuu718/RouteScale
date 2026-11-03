# 本机Serving交付运行手册

范围和验收条件见[交付协议](delivery_protocol.md)。服务使用已验收的Mini-SGLang
I3精度与生命周期补丁，HTTP增加监控，默认只监听127.0.0.1:1969。实验性能和数值
范围仍见[I3报告](../results/i_minisglang/i3_online_v2/README.md)。

## 依赖、源码与模型

使用Python3.11.16，在独立环境安装锁定依赖，不使用项目训练依赖覆盖serving环境：

```bash
uv venv --python 3.11.16 runs/envs/mini-sglang
uv pip sync --torch-backend cu129 --python runs/envs/mini-sglang/bin/python config/delivery/requirements.txt
uv pip check --python runs/envs/mini-sglang/bin/python
python3 scripts/prepare_delivery.py
```

prepare脚本克隆独立上游Git仓库，固定base commit并重放I1七补丁和I2两补丁，验证
干净tree `8ba75c4d14eb6cc985a254fd5185606d62da9061`。离线时传
`--seed /absolute/path/to/preserved/mini-sglang`。已存在的目标只校验，不覆盖。
Python入口直接导入此重放树，不需要向环境editable安装另一份上游。

Qwen3-0.6B模型revision为`c1899de289a04d12100db370d81485cdf75e47ca`，
放在`runs/models/Qwen3-0.6B`。可用环境中的huggingface_hub下载：

```bash
runs/envs/mini-sglang/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='runs/models/Qwen3-0.6B')
PY
```

模型文件与[model.json](../config/delivery/model.json)逐文件核对。首次JIT需要
CUDA13.0工具链位于`runs/toolchains/cuda-13.0-i1`；沿用
`scripts/prepare_i1_toolchain.py`与`scripts/prepare_i1_lifecycle.py`记录的隔离工具链。
本轮依赖重装与GPU验证的精确环境证据保存在最终交付报告，不将容器构建宣称为已验收。
模型、CUDA工具链和依赖必须在干净源码副本外另行恢复；它们不包含在Git里。

## 启动、检查与关闭

```bash
python3 scripts/routeserve.py doctor
python3 scripts/routeserve.py run --run-dir runs/serving/local001
# 另一个终端：
curl -f http://127.0.0.1:1969/live
curl -f http://127.0.0.1:1969/ready
curl -f http://127.0.0.1:1969/metrics
python3 scripts/routeserve.py status --run-dir runs/serving/local001
python3 scripts/routeserve.py stop --run-dir runs/serving/local001
```

run在前台管理独立API/worker进程组。也可Ctrl-C；状态、doctor、命令、API日志和
事件分别在run目录保存，目录必须全新。`stop`核对PID启动标识，停止已结束实例为
幂等操作。默认GPU必须空闲且无另一个runner持锁。若worker死亡，readiness返回
503，监督进程停止服务并返回非零；排查对应日志后用新目录重新启动。

API支持`/v1/models`、`/v1/chat/completions`的文本/流式响应；只使用已实现的
model、messages或prompt、max_tokens、temperature、top_k、top_p、ignore_eos、stream。
示例：

```bash
curl -N http://127.0.0.1:1969/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Tell a short story."}],"max_tokens":32,"temperature":0,"stream":true}'
```

上游usage字段仍为占位零，不用它统计token或计费；取消/错误以客户端终态和
后端结束/资源事件核对。这里没有把未实现的OpenAI参数或并发容量宣称为兼容。
接入总数8、运行数4、请求20秒期限；429带Retry-After，取消后在后端释放前仍占接入位。

## 监控与CPU CI

`/metrics`输出Prometheus文本：HTTP次数、传输耗时总和、HTTP活跃数，以及后端
接入/取消中/前端缓冲/deadline和worker存活gauge。HTTP result是completed、
client_disconnected或error，不等同生成成功。维度仅固定路由/方法/状态，随机路径
归为other；不使用请求ID/prompt作为标签。`/live`只表示API可响应，接流量须检查ready。
原始scheduler/Graph/页分配与释放记录在events中，主机时间不是CUDA kernel计时。

```bash
python3 -m unittest discover -s tests/delivery -p 'test_*.py' -v
```

GitHub工作流`.github/workflows/serving-cpu.yml`运行同一命令，不下载模型或CUDA包。
TP词表gather的额外CPU张量检查在serving环境运行：

```bash
PYTHONPATH=tests runs/envs/mini-sglang/bin/python -m unittest test_tp_head_gather -v
```

## TP租卡前准备

```bash
python3 scripts/prepare_serving_tp.py --stage plan --directory runs/tp/plan001
python3 scripts/prepare_serving_tp.py --stage collective --world 1 --directory runs/tp/nccl001
python3 scripts/prepare_serving_tp.py --stage local-smoke --directory runs/tp/local001
runs/envs/mini-sglang/bin/python scripts/audit_serving_tp.py --directory runs/tp/local001 --local
```

仅在已授权并备齐同一host至少四张空闲GPU时，执行`--stage qualify`；30份资格完成后
用audit脚本不带`--local`独立签发完整gate，再分别执行`--stage benchmark`和
`--stage trace`。每阶段记录config/source/model/workload/张量哈希，禁止覆盖。
当前代码不租用主机；多卡实际结果、预算及付费许可仍待后期集中处理。

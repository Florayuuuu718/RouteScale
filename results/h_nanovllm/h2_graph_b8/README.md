# H2：batch=8 CUDA Graph 独立对照

2026-10-23，RTX 5060 Laptop（8151 MiB）/ WSL2，Qwen3-0.6B，BF16，TP=1。
三个 case × 两个 variant × 三个独立进程，**18/18 正式进程通过**。
这是数值验收合格子集，不是完整 H2 通过；原 batch=1 Graph 和 Prefix gate 的失败
分别保留在相邻 `h2_graph_pilot/`、`h2_pilot/`，不得用这里的结果替代。
后续两项数值差异已通过[新策略验收](../h2_numerics/README.md)解决；本目录仍是历史
原生路径测量，不能将其吞吐倍率套用到统一 paged Prefill 后的新策略。

## 验收与边界

- [`preflight.json`](preflight.json)：三个 case 的 cold/warm × 四步 teacher-forced
  logits，共 24 个张量对照，最大误差均为 0；阈值未放宽。
- [`run.json`](run.json)：18/18 正式进程完成；每进程一个排除的 warmup pair、五个
  measured cold/warm pair；每波 8 请求、固定输出 64 tokens。
- 两个 variant 均关闭 prefix；共同物理 KV tensor 为 32 blocks / 896 MiB，block=256。
  high/none 是请求形态，不代表本轮启用了 prefix 命中；所有 cached tokens 均为 0。
- 所有请求结束后队列、used blocks、非零引用计数均为 0；无抢占、OOM 或窗口内新增编译。
- [`audit.json`](audit.json)：1,440 个正式请求、92,160 个输出 tokens；720 对 sampled
  output-token hashes 全部一致。raw 聚合与 summary 均精确重算通过，源码与权重哈希匹配。
  这仍不是完整全词表 64-step 数值证明或模型质量评价。

## 本机结果

每格为三个独立进程统计量的中位数；吞吐单位 output tokens/s，TPOT 单位 ms/token。
完整 TTFT/TPOT/E2E p50/p95/p99 与内存见 [`summary.json`](summary.json)，逐请求计时见
`raw/<case>/<variant>/run{1,2,3}.json`。

| prompt / prefix 形态 | phase | eager 吞吐 | Graph 吞吐 | 吞吐倍率 | eager / Graph TPOT p50 |
|---|---|---:|---:|---:|---:|
| 128 / high | cold | 132.51 | 1116.79 | 8.43× | 58.69 / 6.25 |
| 128 / high | warm | 133.54 | 1136.15 | 8.51× | 59.69 / 6.19 |
| 512 / none | cold | 129.09 | 847.47 | 6.57× | 58.69 / 7.41 |
| 512 / none | warm | 129.32 | 838.22 | 6.48× | 60.16 / 7.54 |
| 512 / high | cold | 134.05 | 831.91 | 6.21× | 59.16 / 7.70 |
| 512 / high | warm | 131.06 | 822.52 | 6.28× | 60.54 / 7.65 |

Graph 峰值 allocated 比共同对照高 **8.144 MiB**，不能把这等同于 reserved memory 或
nvidia-smi 变化。初始化不计入上表：eager 6.31～7.54 秒，Graph 11.08～12.78 秒；
其中 Graph capture 5.04～6.32 秒，是额外一次性成本。
launcher 固定 `OMP_NUM_THREADS=4`；这些是该主机/环境下的结果，不是生产服务容量评估。

128-token TTFT p50 为 eager 68.06/71.86 ms、Graph 65.05/62.34 ms（cold/warm）；
512-token 各点约 137 ms，两路径基本相当。收益集中在持续生成而非首 token。
这与上游 Graph 仅捕获 Decode model forward、Prefill 保持 eager 的实现边界一致。
未采集本轮 profiler，不能把全部倍率定量归因于某一种 kernel/dispatch 成本。

倍率仅适用于本机、小模型、batch=8、固定 64-token 输出的离线主机可见测量；不能
外推到 batch=1、多卡、生产在线 SLO、大模型，或 RouteScale 原 Dense 权重的加速。
这也不是“训练变快”或“模型质量变好”的证据。

## 复现

```bash
# 已有结果时仅续跑同一 gate 下的缺项；首次运行省略 --resume
runs/envs/nano-vllm/bin/python scripts/run_h2_nanovllm.py --config config/serving/h2_graph_b8.json --resume
.venv/bin/python scripts/audit_h2_nanovllm.py
```

固定版本、输入、计时边界、adapter 与失败 gate 的诊断见
[`docs/h2_ablation_protocol.md`](../../../docs/h2_ablation_protocol.md)。
CPU 回归：`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -v`，
86 项中 84 项通过、2 项 CUDA/Triton 专项按环境设置跳过。GPU 数值验收在独立 serving
环境另行完成，不计入这两项跳过的训练侧单元测试。

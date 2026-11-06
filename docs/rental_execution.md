# F / H2 / I2 集中多卡执行

本文件补充已经冻结的数值协议，只增加部署、串行编排和结果汇总。
已有本机 GPU 结果不重跑、不改写；合成数据测试只验证汇总器，不产生多卡性能结论。

## 实验与阶段对应

| 阶段 | 本次同主机实验 | 规模与资格 | 输出 |
|---|---|---|---|
| H2：nano-vLLM | eager/no-prefix、Graph/no-prefix、eager/prefix，各 TP1/2/4 | Qwen3-0.6B；9 项数值资格、27 次独立性能运行、9 组独立 trace | 同机 TP1 内部全词表对照，吞吐、各 rank 显存、NCCL kernel 时间 |
| I2：Mini-SGLang | Naive、Radix、chunk256/128、Overlap、Graph124/13，各 TP1/2/4 | 同一 Qwen 模型，upper16 精度；21 项资格、63 次性能运行、21 组 trace | HF FP32 原门槛及同机 TP1 对照，吞吐、显存、通信 |
| F：Expert Parallel | EP1/2 FP32/BF16 模块及小 GPT；完整模型单卡/双卡更新；单卡 reference、双卡全专家复制、双卡 EP | 8 项模块/GPT 资格、5 项完整训练启动任务（含 CPU tiny 与 CUDA tiny）；三方案各三轮，共 9 次性能运行 | 全局 16,384 tokens/update；梯度/更新/AdamW 一致性、吞吐、各 rank 显存；独立两卡模块 trace |
| 公共前置 | 三套 Torch 栈 GPU 支持检查；NCCL world1/2/4；拓扑、版本、数据与源码哈希 | 同机四张同型号空闲 GPU | 环境记录、collective 正确性、过程日志、退出与显存释放 |

C1–C7 的 DDP、ZeRO、FSDP2 已有四卡验收，本轮无需重跑。
I3 在线负载、H3 RouteScale 模型接入和 M5 完整 SGLang 交付的现有验收属于各自本机范围。
本次 TP 是固定 continuation 的 scheduler replay，并非多卡 HTTP 生命周期或在线 SLO。
F 的复制对照是显式梯度归约，不是原生 PyTorch DDP reducer 性能。
F trace 是 D/E 形状的 MoE 模块，不把它当作完整 GPT 的通信比例。

教程还提到 I2 较大模型通信/计算对照；当前冻结矩阵只含 Qwen3-0.6B。
较大模型需要另定模型 revision、显存、数值资格和时间预算，不在此次已准备矩阵中。
原生 DDP reducer MoE、多卡在线故障测试、完整 GPT trace 同样属于后续扩展。
J/M7 联合复盘和个人面试材料仍在实验完成后统一处理。

## 部署包与环境

源码提交后，在本机创建全新包目录：

```bash
python3 scripts/prepare_rental.py package --bundle runs/rental/bundle001
python3 scripts/prepare_rental.py verify --bundle runs/rental/bundle001
```

包包含项目和两份 vendor 的 Git bundle、原始 Qwen 文件、TinyStories train/val、
固定 FlashAttention wheel，以及解引用后的 CUDA13.0 工具链。逐文件记录 SHA256；
不包含虚拟环境、个人文档或凭据。包中的 Git 历史仍只包含项目原来跟踪的文件。
CUDA overlay 原来指向 `/usr/local` 的绝对链接已复制为实体文件。
模型和工具链受各自许可证约束，包保留原许可证文件。

传输时使用整个包目录（例如 rsync），在目标 Linux x86_64 主机重新验证。
目标须具备 Git、Python3、uv、兼容 NVIDIA 驱动、足够内存和磁盘。
四卡数量不等于兼容性通过；实际兼容性由三套 Torch 栈和数值资格检查决定。
Python3.11.16 和依赖通过 uv 重建；需要依赖源网络或已有 uv 缓存。
这不是包含所有 Python wheels 的完全离线镜像。

若目标还没有 RouteScale，先用包内源码启动恢复工具：

```bash
git clone /workspace/bundle001/project.bundle /workspace/RouteScale-bootstrap
python3 /workspace/RouteScale-bootstrap/scripts/prepare_rental.py restore \
  --bundle /workspace/bundle001 --root /workspace/RouteScale
cd /workspace/RouteScale
python3 scripts/prepare_rental.py install
python3 scripts/prepare_rental.py check-env
```

`restore` 只接受不存在的目录，重建独立 vendor Git 仓库，避免复制 worktree 的
外部 `.git` 指针。`install` 拒绝覆盖环境，失败日志保留在 `runs/rental/install`；
排查后使用新的恢复目录，不清理旧证据。已有完整 uv 缓存时可使用 `install --offline`。
三套环境分别为训练 `.venv`、nano-vLLM 和 Mini-SGLang，不混装。
训练依赖锁包含 Torch 要求的 CUDA extras；nano 的 FlashAttention 使用单独 SHA 固定 wheel。

## 执行、停止与继续

准备计划不访问 GPU，也不会购买或启动云实例：

```bash
python3 scripts/run_rental.py --directory runs/rental/plan001
```

在已选定且已获准使用的主机上，将
[主机模板](../deploy/rental/host.example.json)复制到忽略路径 `runs/rental/host.json`，
填入 hostname、按物理索引排列的四个 GPU UUID、整机每小时费用、币种、最大费用、
最大运行秒数和 `existing_host_confirmed=true`。机器与预算未确定时保留模板为 false。
`max_runtime_s × hourly_rate_all_gpus / 3600` 必须不超过费用上限。

```bash
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
python3 scripts/run_rental.py --directory runs/rental/host001 \
  --host-config runs/rental/host.json --execute
python3 scripts/summarize_rental.py --directory runs/rental/host001 \
  --output runs/rental/host001-report
```

顺序是主机检查 → NCCL1/2/4 → TP 全资格与独立审计 → F 模块及完整训练资格 →
TP/F 无 Profiler 计时 → 独立 trace。任一失败、OOM、非空闲 GPU、超时或预算耗尽即停止。
每个旧启动器的数值准入继续生效；源文件变化会拒绝后续运行。
总超时上限是故障预算，不能作为实际租用时长预估。

如需资格完成后先检查结果，首轮加 `--through qualification`，随后对同目录加
`--resume --through trace`。仅允许继续成功完成的阶段，原主机、日志、源码、环境和
费用配置必须一致；截止时间从第一次启动算起，包括暂停时间，继续不会重置预算。
失败的半轮不会自动重试或覆盖，需检查保留的日志后建立新实验目录。

控制器跟踪 PID 启动标识及被重新托管的子孙进程，包括自行创建 session 的 worker。
超时/中断先终止、再有限升级清理；清理发生本身记为失败。不会按名称杀其他实验进程。
运行过程保持串行，不能并行跑相互干扰的 GPU 基准。

**停止实验不等于停止计费。** 工具不调用云资源 API；安装、数据传输和导出期间也可能
收费。应把这些时间计入租用总预算；汇出证据并核对进程与显存释放后，另行关闭租用实例。

## 汇总口径与证据回传

[汇总入口](../scripts/summarize_rental.py)核对完整阶段、日志和资格哈希，逐个核对 rank、
请求、wave、输出 token、资源释放、warmup 排除、训练全局窗口和 100 个实测 update。
TP 对每个对齐 wave 取最慢 rank 时间，全局请求只计一次；F 已由 worker 记录 rank-max。
以每个独立进程为样本，输出三次运行的中位数、范围、CV 和配对 TP1/reference 比值。
不把单进程内五个 wave 当成五次独立复验，也不把 scheduler 时间写成 HTTP TTFT。

trace 按每 rank CUDA kernel 的区间并集统计 NCCL、其他 kernel 和重叠区间。
其他 kernel 包含辅助工作，不能全部称为模型计算；Profiler 时间不混入无 Profiler 吞吐。
F 保留审计器的 send matrix 与逻辑流量，逻辑字节不等于网卡实测流量。

输出 `summary.json`、`tp_processes.csv`、`training_processes.csv` 和证据哈希索引。
缺失或失败的会话仍输出拒绝报告，不能生成已验收的性能排名。
汇总应在原主机运行，因为原 F gate 会检查 hostname 和绝对 raw tensor 路径。
将原始 session、汇总目录和部署 manifest 一起回传本机；大张量和 trace 留忽略路径，
仅提交精简报告与哈希。最终结论只覆盖实际完成的矩阵，不自动扩大到 F/H/I 全部扩展任务。

本机验收命令：

```bash
python3 -m unittest discover -s tests/rental -v
runs/envs/delivery-cpu-ci/bin/python -m unittest discover -s tests/delivery -v
python3 scripts/check_delivery_cpu.py
```

部署验收还必须实际完成一次新目录恢复、三环境重建、版本/资产核对和恢复环境下的
CPU Gloo2 tiny 完整训练更新。它证明部署流程和 CPU 路径，不替代目标四卡数值验收。

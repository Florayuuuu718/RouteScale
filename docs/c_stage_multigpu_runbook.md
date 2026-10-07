# C 阶段四卡租用与运行手册

## 租什么机器

租一台物理机内的 4 张同型号 NVIDIA GPU，不要先租跨机器的 4 卡。当前 50.91M 模型本身在 8 GiB 卡上可运行，但 C6 容量边界需要更大显存：预算优先可选 4×RTX 4090 24 GiB；如果重点是 NVLink、较大模型和更稳定的 datacenter 对照，优先 4×A100 40/80 GiB。两种机器的互连不同，不能把一台的 scaling 数字当成另一台的结论。

建议同时满足：

- 4 卡型号和显存完全一致，位于同一节点；
- 至少 128 GiB 系统内存、16 vCPU、100 GiB 可用本地 SSD；
- NVIDIA driver 能运行本项目锁定的 CUDA 12.9 PyTorch wheel；
- 若要运行 ZeRO offload、CPUAdam 或其他 JIT op，镜像中还要有可用的 CUDA Toolkit/`nvcc` 和编译工具；
- 租用时长先买 1–2 小时做准入和 smoke，通过后再续正式矩阵，避免把环境排错时间当实验成本。

## 上机后的准入顺序

```bash
git clone <repository-url> RouteScale
cd RouteScale
uv sync --python 3.11 --extra distributed

# 放置或生成 data/tinystories_full 后，先验证四卡、数据和 NCCL collective。
uv run torchrun --standalone --nproc_per_node=4 scripts/preflight_multigpu.py \
  --expected-gpus=4 \
  --data-dir=data/tinystories_full \
  --output=results/c0_environment/preflight_4gpu.json \
  --strict
```

只有 `failed_required_checks` 为空才继续。`active_warnings` 中若有工作区不干净，需要提交或记录 diff；若 `nvcc` 缺失，可以先跑不需要 JIT 扩展的基线，但不能跑 offload/CPUAdam 后仍声称环境完整。保留 JSON 内的 GPU UUID、driver、PyTorch/CUDA/NCCL、拓扑、数据 SHA256 和 collective 结果。

随后执行测试和真实 CUDA/NCCL 更新对照：

```bash
uv run python -m unittest discover -s tests -v
uv run torchrun --standalone --nproc_per_node=2 scripts/check_ddp_update.py \
  --device=cuda --output=results/c_ddp/correctness/2proc_cuda.json
uv run torchrun --standalone --nproc_per_node=4 scripts/check_ddp_update.py \
  --device=cuda --output=results/c_ddp/correctness/4proc_cuda.json
```

## C5 四卡覆盖 smoke

先用 debug 数据验证 4 rank checkpoint 生命周期；`global_windows_per_update=32`、`batch_size=8` 时每个 rank 每次更新正好处理 8 个窗口。

```bash
uv run torchrun --standalone --nproc_per_node=4 train_coverage.py \
  --data-dir=data/tinystories_debug \
  --out-dir=results/c5_coverage/4gpu_smoke \
  --max-updates=5 --checkpoint-every=2

uv run torchrun --standalone --nproc_per_node=4 train_coverage.py \
  --data-dir=data/tinystories_debug \
  --out-dir=results/c5_coverage/4gpu_smoke \
  --resume-from=results/c5_coverage/4gpu_smoke/checkpoint.pt \
  --max-updates=8 --checkpoint-every=2
```

正式完整覆盖时改用 `tinystories_full`、`--max-updates=0 --full-validation`。正式记录应保留覆盖率、补齐数、训练/validation 尾部、窗口哈希、初末 loss、checkpoint 哈希和每卡峰值显存。

## C4 多 rank 通信 trace

正式 strong/weak scaling 稳定后，单独采集一次四卡 Profiler trace。每个
rank 会写独立的 Chrome trace、key averages 和 metadata；rank 0 另外写
manifest。Profiler 有显著开销，结果只用于解释通信机制，不能替代无
Profiler 吞吐。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN \
python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  train.py config/profile_ddp_tinystories.py \
  --profiler_run_id=4gpu_autodl \
  --profiler_trace_dir=logs/c_ddp_autodl_profiler \
  --profiler_results_dir=results/c_ddp_autodl/profiler

python3 scripts/summarize_c4_profiler.py \
  --results-dir=results/c_ddp_autodl/profiler \
  --run-id=4gpu_autodl \
  --expected-ranks=4
```

`*_summary.json` 会按 rank 汇总 NCCL/AllReduce 等事件。计算与通信是否重叠
必须在 Perfetto/Chrome trace 中检查 backward 与 collective 的时间区间；
key averages 丢失了并发关系，不能单独裁决重叠。

## DeepSpeed 逐级 smoke

必须从 ZeRO-0 开始，确认 loss 和窗口后再逐级增加分片：

```bash
for stage in 0 1 2 3; do
  uv run torchrun --standalone --nproc_per_node=4 train_deepspeed.py \
    --deepspeed-config="config/deepspeed/zero${stage}.json" \
    --data-dir=data/tinystories_debug \
    --out-dir="results/c6_deepspeed/zero${stage}_4gpu_smoke" \
    --max-updates=5 --checkpoint-every=2
done
```

每次结果中的 `resolved_deepspeed_config.json` 才是实际运行配置。恢复时传 checkpoint 根目录与 tag，例如：

```bash
uv run torchrun --standalone --nproc_per_node=4 train_deepspeed.py \
  --deepspeed-config=config/deepspeed/zero3.json \
  --data-dir=data/tinystories_debug \
  --out-dir=results/c6_deepspeed/zero3_4gpu_smoke \
  --resume-dir=results/c6_deepspeed/zero3_4gpu_smoke/checkpoints \
  --resume-tag=update-00000004 --max-updates=8
```

`zero3_offload.json` 是独立变量，只在无 offload 的 ZeRO-3 正确后运行。单卡 stage 1/2/3 smoke 只证明代码路径，四卡结果才证明真实状态分片和通信。

当前 50.91M 的同初始化阶段对照保持默认 `--no-zero-init`。搜索接近 OOM 的 ZeRO-3 容量边界时才显式增加 `--zero-init`，否则模型构造阶段会短暂在每个 rank 上保留完整参数。

ZeRO-2/3 保存后还要验证合并权重能离开 DeepSpeed Engine 使用：

```bash
uv run python scripts/verify_deepspeed_fp32.py \
  --checkpoint-dir=results/c6_deepspeed/zero3_4gpu_smoke/checkpoints \
  --tag=update-00000004 \
  --result-json=results/c6_deepspeed/zero3_4gpu_smoke/result.json \
  --merged-output=results/c6_deepspeed/zero3_4gpu_smoke/merged_fp32.pt \
  --output=results/c6_deepspeed/zero3_4gpu_smoke/fp32_verification.json
```

## FSDP2 smoke

```bash
uv run torchrun --standalone --nproc_per_node=4 train_fsdp2.py \
  --data-dir=data/tinystories_debug \
  --out-dir=results/c7_fsdp2/4gpu_smoke \
  --max-updates=5 --checkpoint-every=2

uv run torchrun --standalone --nproc_per_node=4 train_fsdp2.py \
  --data-dir=data/tinystories_debug \
  --out-dir=results/c7_fsdp2/4gpu_smoke \
  --resume-from=results/c7_fsdp2/4gpu_smoke/checkpoints/update-00000004 \
  --max-updates=8
```

这里验证的是 block 级 `fully_shard`、梯度累积同步边界和 DCP 恢复。不要用 smoke 的总耗时报告正式 speedup。

## C6/C7 正式后端 benchmark

`train_deepspeed.py` 和 `train_fsdp2.py` 的 `--benchmark` 模式与 smoke
生命周期互斥：它只执行预热和正式测量，不运行 validation，也不保存或
恢复 checkpoint。每个 rank 用 CUDA Event 记录更新时间，rank 0 使用每一步
最慢 rank 的时间计算吞吐。默认协议是 20 次预热、100 次测量。

DeepSpeed 对每个 ZeRO stage、每个 GPU 数和每个 run ID 启动新进程。例如
四卡 ZeRO-2：

```bash
for run_id in run1 run2 run3; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN \
  python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_deepspeed.py \
    --deepspeed-config=config/deepspeed/zero2.json \
    --data-dir=data/tinystories_full \
    --tail-policy=drop \
    --fused-optimizer \
    --benchmark \
    --benchmark-run-id="$run_id" \
    --benchmark-results-dir=results/c_backend_autodl
done
```

一卡也必须由 `torchrun --nproc_per_node=1` 启动；两卡固定选择同 NUMA 的
物理 GPU 0、1。对 ZeRO-0/1/2/3 重复 1/2/4 卡矩阵。FSDP2 使用同一参数：

```bash
for run_id in run1 run2 run3; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN \
  python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_fsdp2.py \
    --data-dir=data/tinystories_full \
    --tail-policy=drop \
    --fused-optimizer \
    --benchmark \
    --benchmark-run-id="$run_id" \
    --benchmark-results-dir=results/c_backend_autodl
done
```

全部运行完成后校验 loss、窗口、重复次数，并与同机 DDP strong summary 对照：

```bash
python3 scripts/summarize_c_backends.py \
  --results-dir=results/c_backend_autodl \
  --world-sizes 1 2 4 \
  --variants deepspeed_zero0 deepspeed_zero1 deepspeed_zero2 deepspeed_zero3 fsdp2 \
  --ddp-summary=results/c_ddp_autodl/strong/summary.json
```

这里得到的是当前 50.91M 模型的正式性能点。容量边界实验仍应单独改变
`n_layer`/`n_embd`；ZeRO-3 容量搜索显式使用 `--zero-init`，不得和性能矩阵
默认的 `--no-zero-init` 混在同一汇总中。

## C6/C7 后端通信 trace

正式吞吐矩阵完成后，用统一脚本顺序采集 ZeRO-0/1/2/3 和 FSDP2 的四卡
短 trace。脚本使用 2 次不采样预热，以及 wait=1、profiler warmup=1、
active=3 的 schedule；每个 rank 单独保存 Chrome trace、key averages 和
metadata，最后自动生成 collective-family summary。

```bash
bash scripts/run_c_backend_profiler.sh
```

结果位于 `results/c_backend_profiler/`。重点核对 AllReduce、ReduceScatter、
AllGather 的出现位置；wrapper 和 NCCL kernel 是同一通信工作的不同层级，
不能把二者时间直接相加。是否与 backward 重叠仍需打开 Chrome trace 检查。
Profiler 结果不得替代正式无 Profiler 吞吐。

原始 Chrome trace 仍在服务器时，可统一计算 NCCL kernel 与非 NCCL GPU
kernel 的时间区间交集：

```bash
bash scripts/run_c_trace_overlap.sh
```

结果写入 `results/c_trace_overlap/summary.json`。`direct overlap` 可在后端间
对照；`within backward scope` 只有入口实际写入同名 `record_function` 时才
有意义，scope 缺失得到的 0 不能解释成“通信不在 backward”。

## C6/C7 checkpoint 定量记录

下面的脚本在 50.91M debug 数据上为 ZeRO-2、ZeRO-3 和 FSDP2 分别执行
2 步保存、由新进程恢复到第 3 步，并对 ZeRO-2/3 合并普通 FP32 权重。
结果记录分片数、总字节数、最慢 rank 保存/恢复时间、进程峰值 RSS、合并
时间和严格加载结果。

```bash
bash scripts/run_c_checkpoint_metrics.sh
```

汇总位于 `results/c_checkpoint_metrics/summary.json`。合并后的 `.pt` 和真实
checkpoint 只留在数据盘，不纳入小型证据包。

## C6/C7 容量封板与 FSDP2 可选点

容量成功点由 benchmark JSON 自动发现；显式 OOM 使用结构化 failure record。
例如 ZeRO-2 在 L32/D2304 的 backward gradient bucket OOM：

```bash
python3 scripts/record_capacity_failure.py \
  --variant=deepspeed_zero2 \
  --run-id=L32_D2304 \
  --parameter-count=2155553280 \
  --n-layer=32 --n-head=36 --n-embd=2304 \
  --failure-type=cuda_oom \
  --phase='backward gradient bucket flatten/allreduce-scatter' \
  --gpu-index=3 \
  --torch-allocated-mib=18698 \
  --torch-reserved-unallocated-mib=4710 \
  --requested-allocation-mib=922 \
  --log=results/c_capacity/logs/zero2_L32_D2304.log \
  --output=results/c_capacity/failures/zero2_L32_D2304.json
```

资源数值必须来自对应 OOM 行或同次运行的监控；无法可靠取得的字段直接省略，
summary 会保留为 `null`，不得用邻近成功点代填。

FSDP2 在 DDP OOM 的 9.84 亿参数点使用相同短容量协议：

```bash
set +e
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN \
timeout --signal=INT --kill-after=30s 300s \
python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  train_fsdp2.py \
  --data-dir=data/tinystories_full \
  --tail-policy=drop \
  --n-layer=32 --n-head=24 --n-embd=1536 \
  --fused-optimizer \
  --benchmark \
  --benchmark-warmup-updates=2 \
  --benchmark-measure-updates=5 \
  --benchmark-results-dir=results/c_capacity/backend \
  --benchmark-run-id=L32_D1536 \
  2>&1 | tee results/c_capacity/logs/fsdp2_L32_D1536.log
echo "FSDP2_L32_D1536_EXIT=${PIPESTATUS[0]}"
set -e

python3 scripts/summarize_c_capacity.py \
  --results-dir=results/c_capacity
```

若 ZeRO-3 没有继续搜索到失败点，汇总必须将最大成功点写成已验证下界，
不得声称为精确最大容量。

## 小型结果证据包

关机前将 JSON、summary、key averages 和日志打包；脚本自动排除数据、
checkpoint、合并权重、runtime 目录和大型 Chrome trace：

```bash
python3 scripts/package_c_results.py \
  --results-dir=results \
  --output=/root/autodl-tmp/routescale_c_evidence.tar.gz

sha256sum /root/autodl-tmp/routescale_c_evidence.tar.gz
```

## 什么时候开始正式矩阵

四卡 preflight、2/4 卡 DDP 更新、C5 恢复、ZeRO-0、ZeRO-1/2/3、FSDP2 smoke 全部通过后，再冻结 commit、数据 SHA256、模型、BF16、全局 tokens/update 和 optimizer。正式性能矩阵使用“20 次预热、100 次测量、3 个独立进程”的 `--benchmark` 模式，并把 Profiler、checkpoint 和 validation 排除在稳态计时外。smoke 的 `elapsed_seconds_this_invocation` 仍不能作为正式 benchmark 裁决值。

本轮四卡实验已于 2026-10-07 完成，本手册保留为复现流程。最终数值、两次
实例拓扑差异、容量夹逼和后端选择见 [`c_stage_results.md`](c_stage_results.md)；
经过 SHA256 清单校验的小型证据见 `artifacts/c_stage_autodl/`。

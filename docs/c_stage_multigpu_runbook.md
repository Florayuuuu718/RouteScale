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

## 什么时候开始正式矩阵

四卡 preflight、2/4 卡 DDP 更新、C5 恢复、ZeRO-0、ZeRO-1/2/3、FSDP2 smoke 全部通过后，再冻结 commit、数据 SHA256、模型、BF16、全局 tokens/update 和 optimizer。正式性能矩阵仍使用“20 次预热、100 次测量、3 个独立进程”的协议，并把 Profiler、checkpoint 和 validation 排除在稳态计时外。当前这些入口已经具备训练和 checkpoint 骨架，但 smoke 的 `elapsed_seconds_this_invocation` 不是最终的 20/100×3 benchmark 裁决值。

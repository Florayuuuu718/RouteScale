# RouteScale 环境

在仓库根目录执行：

```bash
uv sync --python 3.11
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

首次运行 `uv sync` 会生成 `uv.lock` 并创建独立的 `.venv`；以后按锁文件同步。PyTorch 使用 CUDA 12.9 轮子；其他依赖使用默认包索引。需要加载 Hugging Face 模型、用 Weights & Biases 记录或运行 DeepSpeed 时，分别运行 `uv sync --extra hf`、`uv sync --extra logging` 或 `uv sync --extra distributed`。分布式 extra 当前锁定 DeepSpeed 0.19.7，并使用传入的 PyTorch AdamW，基础 ZeRO smoke 不要求编译 fused optimizer。

当前验证设备为 RTX 5060 Laptop GPU（单卡，约 8 GiB）。正式的 2/4 卡 DDP 实验需要多卡机器。nanoGPT 自带的 Shakespeare 数据准备脚本需要联网下载数据；本环境包含其所需的 `requests`。

本机没有 `nvcc`。这不影响当前原生 DDP、FSDP2 和传入 PyTorch optimizer 的 DeepSpeed ZeRO-0/1/2/3 smoke，但不应把它外推为所有 DeepSpeed JIT/offload op 都可用。租用服务器后先运行：

```bash
uv sync --extra distributed
uv run torchrun --standalone --nproc_per_node=4 scripts/preflight_multigpu.py \
  --expected-gpus=4 --data-dir=data/tinystories_full --strict
```

机器规格、准入条件和后续命令见 [`c_stage_multigpu_runbook.md`](c_stage_multigpu_runbook.md)。

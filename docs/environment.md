# RouteScale 环境

在仓库根目录执行：

```bash
uv sync --python 3.11
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

首次运行 `uv sync` 会生成 `uv.lock` 并创建独立的 `.venv`；以后按锁文件同步。PyTorch 使用 CUDA 12.9 轮子；其他依赖使用默认包索引。需要加载 Hugging Face 模型或用 Weights & Biases 记录时，分别运行 `uv sync --extra hf` 或 `uv sync --extra logging`。

当前验证设备为 RTX 5060 Laptop GPU（单卡，约 8 GiB）。正式的 2/4 卡 DDP 实验需要多卡机器。nanoGPT 自带的 Shakespeare 数据准备脚本需要联网下载数据；本环境包含其所需的 `requests`。

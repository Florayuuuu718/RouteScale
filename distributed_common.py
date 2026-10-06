"""Shared deterministic training primitives for RouteScale stage C.

The native DDP, DeepSpeed and FSDP2 entry points intentionally share this
module.  That keeps model construction, optimizer parameter groups, data
windows and validation batches identical while the distributed backend varies.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from model import GPT, GPTConfig


@dataclass(frozen=True)
class DistributedEnvironment:
    rank: int
    local_rank: int
    world_size: int
    device: str
    device_type: str
    distributed: bool

    @property
    def master(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    *, device: str = "auto", backend: str = "auto"
) -> DistributedEnvironment:
    """Initialize a torchrun process group and select the rank-local device."""

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched = "RANK" in os.environ

    if device == "auto":
        selected_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        selected_device = f"cuda:{local_rank}"
    elif device.startswith("cuda:"):
        selected_device = device
    elif device == "cpu":
        selected_device = "cpu"
    else:
        raise ValueError(f"unsupported device: {device}")

    device_type = "cuda" if selected_device.startswith("cuda") else "cpu"
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(selected_device)

    if launched and not dist.is_initialized():
        selected_backend = (
            "nccl" if backend == "auto" and device_type == "cuda" else backend
        )
        if selected_backend == "auto":
            selected_backend = "gloo"
        init_kwargs: dict[str, Any] = {"backend": selected_backend}
        if selected_backend == "nccl":
            init_kwargs["device_id"] = torch.device(selected_device)
        dist.init_process_group(**init_kwargs)

    return DistributedEnvironment(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=selected_device,
        device_type=device_type,
        distributed=dist.is_initialized(),
    )


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"])


def gather_rank_objects(value: Any, env: DistributedEnvironment) -> list[Any] | None:
    if not env.distributed:
        return [value]
    gathered = [None] * env.world_size if env.master else None
    dist.gather_object(value, gathered, dst=0)
    return gathered


class MemmapTokenDataset:
    """Read exact, non-overlapping token windows from nanoGPT ``*.bin`` files."""

    def __init__(self, data_dir: str | Path, block_size: int) -> None:
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.manifest_path = self.data_dir / "manifest.json"
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.train_path = self.data_dir / "train.bin"
        self.val_path = self.data_dir / "val.bin"
        if not self.val_path.exists():
            self.val_path = self.data_dir / "validation.bin"
        for path in (self.train_path, self.val_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._train = np.memmap(self.train_path, dtype=np.uint16, mode="r")
        self._val = np.memmap(self.val_path, dtype=np.uint16, mode="r")
        self.train_window_count = (len(self._train) - 1) // block_size
        self.val_window_count = (len(self._val) - 1) // block_size
        if self.train_window_count < 1 or self.val_window_count < 1:
            raise ValueError("dataset is too short for the selected block_size")

    @property
    def vocab_size(self) -> int:
        return int(self.manifest["vocab_size"])

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.manifest_path)

    def batch(
        self, split: str, window_ids: np.ndarray | list[int], device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._train if split == "train" else self._val
        limit = self.train_window_count if split == "train" else self.val_window_count
        ids = np.asarray(window_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= limit:
            raise ValueError(f"invalid {split} window IDs")
        starts = ids * self.block_size
        x = np.stack(
            [np.asarray(data[i : i + self.block_size], dtype=np.int64) for i in starts]
        )
        y = np.stack(
            [
                np.asarray(data[i + 1 : i + 1 + self.block_size], dtype=np.int64)
                for i in starts
            ]
        )
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        if device.startswith("cuda"):
            x_tensor = x_tensor.pin_memory().to(device, non_blocking=True)
            y_tensor = y_tensor.pin_memory().to(device, non_blocking=True)
        else:
            x_tensor = x_tensor.to(device)
            y_tensor = y_tensor.to(device)
        return x_tensor, y_tensor

    def fixed_validation_ids(self, count: int, seed: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("validation batch count must be positive")
        generator = np.random.Generator(np.random.PCG64(seed))
        return generator.choice(
            self.val_window_count,
            size=count,
            replace=count > self.val_window_count,
        ).astype(np.int64, copy=False)

    @property
    def validation_target_count(self) -> int:
        return len(self._val) - 1

    @property
    def validation_tail_token_count(self) -> int:
        return self.validation_target_count % self.block_size

    def validation_tail_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        length = self.validation_tail_token_count
        if length == 0:
            return None
        start = self.val_window_count * self.block_size
        x = torch.from_numpy(
            np.asarray(self._val[start : start + length], dtype=np.int64)[None, :]
        )
        y = torch.from_numpy(
            np.asarray(self._val[start + 1 : start + 1 + length], dtype=np.int64)[None, :]
        )
        if device.startswith("cuda"):
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)


def build_model(
    *,
    block_size: int,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    dropout: float,
    bias: bool,
) -> GPT:
    return GPT(
        GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
    )


def build_adamw(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    fused: bool = False,
) -> torch.optim.AdamW:
    """Construct the same two AdamW parameter groups for every backend."""

    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    decay = [value for value in parameters.values() if value.dim() >= 2]
    no_decay = [value for value in parameters.values() if value.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    kwargs = {"fused": True} if fused and fused_available else {}
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        **kwargs,
    )


def autocast_context(device_type: str, dtype: str):
    if device_type != "cuda" or dtype == "float32":
        return nullcontext()
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    validation_ids: np.ndarray,
    *,
    batch_size: int,
    device: str,
    device_type: str,
    dtype: str,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for start in range(0, len(validation_ids), batch_size):
        x, y = dataset.batch(
            "val", validation_ids[start : start + batch_size], device
        )
        with autocast_context(device_type, dtype):
            _, loss = model(x, y)
        losses.append(float(loss.detach()))
    if was_training:
        model.train()
    value = torch.tensor(sum(losses) / len(losses), device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return float(value)


@torch.no_grad()
def evaluate_full_validation(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    *,
    batch_size: int,
    device: str,
    device_type: str,
    dtype: str,
) -> float:
    """Evaluate every validation target token, including the short tail."""

    was_training = model.training
    model.eval()
    negative_log_likelihood = 0.0
    target_count = 0
    ids = np.arange(dataset.val_window_count, dtype=np.int64)
    for start in range(0, len(ids), batch_size):
        selected = ids[start : start + batch_size]
        x, y = dataset.batch("val", selected, device)
        with autocast_context(device_type, dtype):
            _, loss = model(x, y)
        tokens = len(selected) * dataset.block_size
        negative_log_likelihood += float(loss.detach()) * tokens
        target_count += tokens
    tail = dataset.validation_tail_batch(device)
    if tail is not None:
        x, y = tail
        with autocast_context(device_type, dtype):
            _, loss = model(x, y)
        tokens = y.numel()
        negative_log_likelihood += float(loss.detach()) * tokens
        target_count += tokens
    if was_training:
        model.train()
    totals = torch.tensor(
        [negative_log_likelihood, float(target_count)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0] / totals[1])


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_json_dump(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, destination)


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
    }


def environment_metadata(env: DistributedEnvironment) -> dict[str, Any]:
    cuda_name = None
    total_memory = None
    if env.device_type == "cuda":
        properties = torch.cuda.get_device_properties(env.device)
        cuda_name = properties.name
        total_memory = properties.total_memory
    return {
        **asdict(env),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": cuda_name,
        "gpu_total_memory_bytes": total_memory,
    }

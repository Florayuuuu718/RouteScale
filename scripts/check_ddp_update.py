"""Compare one sharded DDP update with an equivalent single-model update.

Examples:
  uv run python scripts/check_ddp_update.py --device cuda
  uv run torchrun --standalone --nproc_per_node=2 \
    scripts/check_ddp_update.py --device cpu
  uv run torchrun --standalone --nproc_per_node=2 \
    scripts/check_ddp_update.py --device cuda
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ddp_windows import GlobalWindowScheduler  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual - expected).abs()
    max_abs = difference.max().item() if difference.numel() else 0.0
    scale = expected.abs().max().item() if expected.numel() else 0.0
    relative_to_scale = max_abs / max(scale, 1e-12)
    return max_abs, relative_to_scale


def main() -> None:
    args = parse_args()
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    if args.device == "auto":
        device_kind = (
            "cuda" if torch.cuda.device_count() >= world_size else "cpu"
        )
    else:
        device_kind = args.device
    if device_kind == "cuda" and torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"requested {world_size} CUDA ranks, found {torch.cuda.device_count()} GPUs"
        )
    backend = "nccl" if device_kind == "cuda" else "gloo"
    if distributed:
        if device_kind == "cuda":
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=backend, device_id=torch.device(f"cuda:{local_rank}")
            )
        else:
            dist.init_process_group(backend=backend)

    device = torch.device(
        f"cuda:{local_rank}" if device_kind == "cuda" else "cpu"
    )
    torch.manual_seed(args.seed)
    if device_kind == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=False,
    )
    initial_model = GPT(config).to(device)
    reference_model = copy.deepcopy(initial_model)
    sharded_model = copy.deepcopy(initial_model)
    del initial_model

    reference_optimizer = torch.optim.SGD(
        reference_model.parameters(), lr=args.learning_rate
    )
    if distributed:
        if device_kind == "cuda":
            train_model = DDP(sharded_model, device_ids=[local_rank])
        else:
            train_model = DDP(sharded_model)
    else:
        train_model = sharded_model
    sharded_optimizer = torch.optim.SGD(
        train_model.parameters(), lr=args.learning_rate
    )

    window_count = args.global_batch_size + 5
    local_batch_size = args.global_batch_size // world_size
    scheduler = GlobalWindowScheduler(
        window_count=window_count,
        global_windows_per_update=args.global_batch_size,
        world_size=world_size,
        rank=rank,
        batch_size=local_batch_size,
        seed=args.seed,
        tail_policy="drop",
    )
    global_window_ids = scheduler.global_window_ids(0)
    local_window_ids = scheduler.rank_window_ids(0).reshape(-1)

    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(args.seed + 1)
    token_windows = torch.randint(
        0,
        args.vocab_size,
        (window_count, args.block_size + 1),
        generator=data_generator,
        dtype=torch.long,
    )
    global_tokens = token_windows[torch.from_numpy(global_window_ids)]
    local_tokens = token_windows[torch.from_numpy(local_window_ids)]
    global_x = global_tokens[:, :-1].contiguous().to(device)
    global_y = global_tokens[:, 1:].contiguous().to(device)
    local_x = local_tokens[:, :-1].contiguous().to(device)
    local_y = local_tokens[:, 1:].contiguous().to(device)

    _, reference_loss = reference_model(global_x, global_y)
    reference_loss.backward()
    reference_optimizer.step()

    _, local_loss = train_model(local_x, local_y)
    local_loss.backward()
    sharded_optimizer.step()

    aggregate_loss = local_loss.detach().clone()
    if distributed:
        dist.all_reduce(aggregate_loss, op=dist.ReduceOp.SUM)
        aggregate_loss /= world_size

    parameter_max_abs = 0.0
    parameter_max_relative = 0.0
    reference_scale = 0.0
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in sharded_model.named_parameters():
        reference_parameter = reference_parameters[name]
        max_abs, max_relative = tensor_error(parameter, reference_parameter)
        parameter_max_abs = max(parameter_max_abs, max_abs)
        parameter_max_relative = max(parameter_max_relative, max_relative)
        reference_scale = max(
            reference_scale, reference_parameter.detach().abs().max().item()
        )

    replica_max_abs = 0.0
    if distributed:
        for parameter in sharded_model.parameters():
            rank_zero_parameter = parameter.detach().clone()
            dist.broadcast(rank_zero_parameter, src=0)
            replica_max_abs = max(
                replica_max_abs,
                (parameter.detach() - rank_zero_parameter).abs().max().item(),
            )

    loss_max_abs = abs(aggregate_loss.item() - reference_loss.item())
    loss_scale = abs(reference_loss.item())
    parameter_limit = args.atol + args.rtol * reference_scale
    loss_limit = args.atol + args.rtol * loss_scale
    local_passed = (
        math.isfinite(reference_loss.item())
        and math.isfinite(aggregate_loss.item())
        and parameter_max_abs <= parameter_limit
        and loss_max_abs <= loss_limit
        and replica_max_abs <= args.atol
    )
    pass_tensor = torch.tensor(
        int(local_passed), device=device, dtype=torch.int32
    )
    if distributed:
        dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    passed = bool(pass_tensor.item())

    local_record = {
        "rank": rank,
        "local_window_ids": local_window_ids.tolist(),
        "local_loss": local_loss.item(),
        "parameter_max_abs_error": parameter_max_abs,
        "parameter_max_relative_to_reference_scale": parameter_max_relative,
        "replica_max_abs_error": replica_max_abs,
    }
    if distributed:
        gathered_records = [None] * world_size if rank == 0 else None
        dist.gather_object(local_record, gathered_records, dst=0)
    else:
        gathered_records = [local_record]

    if rank == 0:
        output_path = args.output or (
            REPOSITORY_ROOT
            / "results"
            / "c_ddp"
            / "correctness"
            / f"{world_size}proc_{device_kind}.json"
        )
        result = {
            "schema_version": 1,
            "check": "single global update versus rank-sharded DDP update",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": device_kind,
                "backend": backend if distributed else None,
                "world_size": world_size,
            },
            "configuration": {
                "seed": args.seed,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch_size,
                "block_size": args.block_size,
                "vocab_size": args.vocab_size,
                "optimizer": "SGD",
                "learning_rate": args.learning_rate,
                "gradient_clipping": None,
                "atol": args.atol,
                "rtol": args.rtol,
            },
            "data": {
                "global_window_ids": global_window_ids.tolist(),
                "global_window_ids_sha256": hashlib.sha256(
                    np.asarray(global_window_ids, dtype="<i8").tobytes()
                ).hexdigest(),
            },
            "comparison": {
                "reference_loss": reference_loss.item(),
                "aggregate_sharded_loss": aggregate_loss.item(),
                "loss_max_abs_error": loss_max_abs,
                "loss_limit": loss_limit,
                "parameter_max_abs_error": max(
                    record["parameter_max_abs_error"] for record in gathered_records
                ),
                "parameter_max_relative_to_reference_scale": max(
                    record["parameter_max_relative_to_reference_scale"]
                    for record in gathered_records
                ),
                "parameter_limit": parameter_limit,
                "replica_max_abs_error": max(
                    record["replica_max_abs_error"] for record in gathered_records
                ),
            },
            "per_rank": gathered_records,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"DDP update check {status}: world_size={world_size}, device={device_kind}, "
            f"parameter max abs={result['comparison']['parameter_max_abs_error']:.3e}, "
            f"replica max abs={result['comparison']['replica_max_abs_error']:.3e}"
        )
        print(f"wrote {output_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

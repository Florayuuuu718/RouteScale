"""Shared no-Profiler benchmark driver for C-stage distributed backends."""

from __future__ import annotations

import hashlib
import math
import resource
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Callable

import numpy as np
import torch

from ddp_windows import GlobalWindowScheduler
from distributed_common import (
    DistributedEnvironment,
    MemmapTokenDataset,
    atomic_json_dump,
    barrier,
    environment_metadata,
    gather_rank_objects,
    git_metadata,
)


def add_benchmark_arguments(parser, *, default_results_dir: str) -> None:
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run a no-validation, no-checkpoint formal performance benchmark",
    )
    parser.add_argument("--benchmark-warmup-updates", type=int, default=20)
    parser.add_argument("--benchmark-measure-updates", type=int, default=100)
    parser.add_argument("--benchmark-run-id", default="run1")
    parser.add_argument("--benchmark-results-dir", default=default_results_dir)


def validate_benchmark_arguments(args, scheduler: GlobalWindowScheduler) -> None:
    if args.benchmark_warmup_updates <= 0:
        raise ValueError("benchmark-warmup-updates must be positive")
    if args.benchmark_measure_updates <= 0:
        raise ValueError("benchmark-measure-updates must be positive")
    total = args.benchmark_warmup_updates + args.benchmark_measure_updates
    if total > scheduler.updates_per_epoch:
        raise ValueError(
            "benchmark warmup plus measurement exceeds one scheduled epoch"
        )


def aggregate_rank_records(
    rank_records: list[dict], *, measured_steps: int, global_tokens_per_update: int
) -> dict:
    """Aggregate synchronized-rank records using the slowest rank per update."""

    if not rank_records:
        raise ValueError("rank_records must not be empty")
    ordered = sorted(rank_records, key=lambda item: item["rank"])
    if any(len(item["step_times_ms"]) != measured_steps for item in ordered):
        raise ValueError("rank record step count does not match measured_steps")
    if any(len(item["mean_micro_batch_losses"]) != measured_steps for item in ordered):
        raise ValueError("rank record loss count does not match measured_steps")

    slowest = [
        max(item["step_times_ms"][step] for item in ordered)
        for step in range(measured_steps)
    ]
    global_losses = [
        mean(item["mean_micro_batch_losses"][step] for item in ordered)
        for step in range(measured_steps)
    ]
    median_ms = median(slowest)
    total_tokens = measured_steps * global_tokens_per_update
    return {
        "per_rank": ordered,
        "slowest_rank_step_times_ms": slowest,
        "global_mean_losses": global_losses,
        "all_losses_finite": all(
            item["all_losses_finite"] for item in ordered
        ) and all(math.isfinite(value) for value in global_losses),
        "all_window_hashes_match_plan": all(
            item["warmup_window_ids_sha256"]
            == item["expected_warmup_window_ids_sha256"]
            and item["measurement_window_ids_sha256"]
            == item["expected_measurement_window_ids_sha256"]
            for item in ordered
        ),
        "median_slowest_rank_step_ms": median_ms,
        "mean_slowest_rank_step_ms": mean(slowest),
        "min_slowest_rank_step_ms": min(slowest),
        "max_slowest_rank_step_ms": max(slowest),
        "global_tokens_per_second_from_median": (
            global_tokens_per_update / (median_ms / 1000)
        ),
        "aggregate_global_tokens_per_second": (
            total_tokens / (sum(slowest) / 1000)
        ),
        "first_loss": global_losses[0],
        "last_loss": global_losses[-1],
    }


def run_distributed_benchmark(
    *,
    args,
    env: DistributedEnvironment,
    dataset: MemmapTokenDataset,
    scheduler: GlobalWindowScheduler,
    backend: str,
    variant: str,
    model_parameter_count: int,
    run_update: Callable[[int], torch.Tensor],
    extra_configuration: dict | None = None,
) -> Path | None:
    """Run warmup and measured updates without validation or checkpoint I/O."""

    if env.device_type != "cuda":
        raise RuntimeError("formal distributed benchmark requires CUDA")
    validate_benchmark_arguments(args, scheduler)

    warmup_steps = args.benchmark_warmup_updates
    measured_steps = args.benchmark_measure_updates
    measurement_start = warmup_steps
    if env.master:
        print(
            f"{variant} benchmark: world_size={env.world_size}, "
            f"{warmup_steps} warmup updates, {measured_steps} measured updates, "
            f"run {args.benchmark_run_id}",
            flush=True,
        )

    warmup_hasher = hashlib.sha256()
    for update in range(warmup_steps):
        warmup_ids = scheduler.rank_window_ids(update).reshape(-1)
        warmup_hasher.update(
            np.asarray(warmup_ids, dtype="<i8").tobytes(order="C")
        )
        loss = run_update(update)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite warmup loss at update {update}")

    torch.cuda.synchronize(env.device)
    barrier()
    torch.cuda.reset_peak_memory_stats(env.device)

    start_events: list[torch.cuda.Event] = []
    end_events: list[torch.cuda.Event] = []
    measured_losses: list[torch.Tensor] = []
    measurement_hasher = hashlib.sha256()
    for measured_step in range(measured_steps):
        update = measurement_start + measured_step
        measured_ids = scheduler.rank_window_ids(update).reshape(-1)
        measurement_hasher.update(
            np.asarray(measured_ids, dtype="<i8").tobytes(order="C")
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = run_update(update)
        end.record()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite measured loss at update {update}")
        start_events.append(start)
        end_events.append(end)
        measured_losses.append(loss.detach().float())

    torch.cuda.synchronize(env.device)
    local_times = [
        start.elapsed_time(end) for start, end in zip(start_events, end_events)
    ]
    local_losses = [float(value) for value in measured_losses]
    local_record = {
        "rank": env.rank,
        "local_rank": env.local_rank,
        "gpu": torch.cuda.get_device_name(env.device),
        "step_times_ms": local_times,
        "mean_micro_batch_losses": local_losses,
        "all_losses_finite": all(math.isfinite(value) for value in local_losses),
        "warmup_window_ids_sha256": warmup_hasher.hexdigest(),
        "measurement_window_ids_sha256": measurement_hasher.hexdigest(),
        "expected_warmup_window_ids_sha256": scheduler.hash_updates(
            0, warmup_steps, env.rank
        ),
        "expected_measurement_window_ids_sha256": scheduler.hash_updates(
            measurement_start, measured_steps, env.rank
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
        "device_total_bytes": torch.cuda.get_device_properties(env.device).total_memory,
        # Linux reports ru_maxrss in KiB. This covers model construction,
        # optimizer creation, warmup, and the measured updates for this rank.
        "peak_process_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    # Peak CUDA values above are already captured. Release unused allocator
    # cache so result collection itself does not become the capacity limit.
    torch.cuda.empty_cache()
    gathered = gather_rank_objects(local_record, env)
    output_path = None
    if env.master:
        global_tokens = args.global_windows_per_update * args.block_size
        measurement = aggregate_rank_records(
            gathered,
            measured_steps=measured_steps,
            global_tokens_per_update=global_tokens,
        )
        measurement.update(
            {
                "timing": (
                    "per-rank CUDA Events; one synchronization after measurement; "
                    "global update uses the slowest rank"
                ),
                "profiler_enabled": False,
                "warmup_steps": warmup_steps,
                "measured_steps": measured_steps,
                "total_measured_tokens": measured_steps * global_tokens,
            }
        )
        measured_global_ids = np.concatenate(
            [
                scheduler.global_window_ids(update)
                for update in range(
                    measurement_start, measurement_start + measured_steps
                )
            ]
        )
        result = {
            "schema_version": 1,
            "benchmark": "C distributed backend no-Profiler benchmark",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "benchmark_complete",
            "backend": backend,
            "variant": variant,
            "run_id": args.benchmark_run_id,
            "environment": environment_metadata(env),
            "git": git_metadata(),
            "data": {
                "data_dir": str(dataset.data_dir),
                "manifest_sha256": dataset.manifest_sha256,
                "train_window_count": dataset.train_window_count,
                "data_seed": args.data_seed,
                "tail_policy": args.tail_policy,
                "warmup_global_window_ids_sha256": scheduler.hash_updates(
                    0, warmup_steps
                ),
                "measurement_global_window_ids_sha256": scheduler.hash_updates(
                    measurement_start, measured_steps
                ),
                "measurement_window_count": len(measured_global_ids),
                "measurement_repeated_window_count": (
                    len(measured_global_ids) - len(np.unique(measured_global_ids))
                ),
            },
            "configuration": {
                **vars(args),
                "world_size": env.world_size,
                "local_micro_steps": scheduler.local_micro_steps,
                "local_windows_per_update": scheduler.local_windows_per_update,
                "local_tokens_per_update": (
                    scheduler.local_windows_per_update * args.block_size
                ),
                "global_tokens_per_update": global_tokens,
                "model_parameter_count": model_parameter_count,
                **(extra_configuration or {}),
            },
            "measurement": measurement,
        }
        output_path = (
            Path(args.benchmark_results_dir)
            / variant
            / f"{env.world_size}gpu"
            / f"{args.benchmark_run_id}.json"
        )
        atomic_json_dump(result, output_path)
        print(
            f"{variant} result: median slowest-rank step "
            f"{measurement['median_slowest_rank_step_ms']:.3f} ms, "
            f"{measurement['global_tokens_per_second_from_median']:,.0f} tokens/s",
            flush=True,
        )
        print(f"saved {output_path}", flush=True)
    barrier()
    return output_path

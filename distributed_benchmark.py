"""Shared benchmark and profiler drivers for C-stage distributed backends."""

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


def add_benchmark_arguments(
    parser,
    *,
    default_results_dir: str,
    default_profiler_dir: str = "results/c_backend_profiler",
) -> None:
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run a no-validation, no-checkpoint formal performance benchmark",
    )
    parser.add_argument("--benchmark-warmup-updates", type=int, default=20)
    parser.add_argument("--benchmark-measure-updates", type=int, default=100)
    parser.add_argument("--benchmark-run-id", default="run1")
    parser.add_argument("--benchmark-results-dir", default=default_results_dir)
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="collect a short per-rank communication trace; not formal timing",
    )
    parser.add_argument("--profiler-startup-warmup-updates", type=int, default=2)
    parser.add_argument("--profiler-wait-updates", type=int, default=1)
    parser.add_argument("--profiler-warmup-updates", type=int, default=1)
    parser.add_argument("--profiler-active-updates", type=int, default=3)
    parser.add_argument("--profiler-run-id", default="4gpu")
    parser.add_argument("--profiler-results-dir", default=default_profiler_dir)
    parser.add_argument(
        "--profiler-trace-dir",
        default=f"{default_profiler_dir}/traces",
    )


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


def validate_profiler_arguments(args, scheduler: GlobalWindowScheduler) -> None:
    values = (
        args.profiler_startup_warmup_updates,
        args.profiler_wait_updates,
        args.profiler_warmup_updates,
        args.profiler_active_updates,
    )
    if min(values) <= 0:
        raise ValueError("profiler update counts must be positive")
    if sum(values) > scheduler.updates_per_epoch:
        raise ValueError("profiler updates exceed one scheduled epoch")


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


def run_distributed_profiler(
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
    """Collect short per-rank traces without treating them as formal timing."""

    if env.device_type != "cuda":
        raise RuntimeError("distributed profiler requires CUDA")
    validate_profiler_arguments(args, scheduler)

    startup_steps = args.profiler_startup_warmup_updates
    wait_steps = args.profiler_wait_updates
    profiler_warmup_steps = args.profiler_warmup_updates
    active_steps = args.profiler_active_updates
    scheduled_steps = wait_steps + profiler_warmup_steps + active_steps
    total_steps = startup_steps + scheduled_steps
    if env.master:
        print(
            f"{variant} profiler: world_size={env.world_size}, "
            f"startup={startup_steps}, wait={wait_steps}, "
            f"warmup={profiler_warmup_steps}, active={active_steps}, "
            f"run {args.profiler_run_id}",
            flush=True,
        )

    window_hasher = hashlib.sha256()
    observed_losses: list[torch.Tensor] = []
    for update in range(startup_steps):
        ids = scheduler.rank_window_ids(update).reshape(-1)
        window_hasher.update(np.asarray(ids, dtype="<i8").tobytes(order="C"))
        loss = run_update(update)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite profiler warmup loss at {update}")
        observed_losses.append(loss.detach().float())

    torch.cuda.synchronize(env.device)
    barrier()
    torch.cuda.reset_peak_memory_stats(env.device)

    results_dir = Path(args.profiler_results_dir) / variant
    trace_dir = Path(args.profiler_trace_dir) / variant
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = f"{args.profiler_run_id}_rank{env.rank}"
    trace_path = trace_dir / f"{artifact_stem}_trace.json"
    table_path = results_dir / f"{artifact_stem}_key_averages.txt"
    events_path = results_dir / f"{artifact_stem}_key_averages.json"

    def trace_handler(prof) -> None:
        prof.export_chrome_trace(str(trace_path))
        averages = list(prof.key_averages())
        table_path.write_text(
            prof.key_averages().table(
                sort_by="self_device_time_total",
                row_limit=100,
                max_name_column_width=100,
            )
            + "\n",
            encoding="utf-8",
        )
        rows = [
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
                "cpu_time_total_us": event.cpu_time_total,
                "self_device_time_us": event.self_device_time_total,
                "device_time_total_us": event.device_time_total,
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "self_device_memory_bytes": event.self_device_memory_usage,
            }
            for event in averages
        ]
        rows.sort(key=lambda event: event["self_device_time_us"], reverse=True)
        atomic_json_dump(rows, events_path)

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        schedule=torch.profiler.schedule(
            wait=wait_steps,
            warmup=profiler_warmup_steps,
            active=active_steps,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for profile_step in range(scheduled_steps):
            update = startup_steps + profile_step
            ids = scheduler.rank_window_ids(update).reshape(-1)
            window_hasher.update(
                np.asarray(ids, dtype="<i8").tobytes(order="C")
            )
            loss = run_update(update)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite profiler loss at update {update}"
                )
            observed_losses.append(loss.detach().float())
            prof.step()

    torch.cuda.synchronize(env.device)
    if not trace_path.is_file() or not events_path.is_file():
        raise RuntimeError("profiler did not emit the expected rank artifacts")
    losses = [float(value) for value in observed_losses]
    expected_hash = scheduler.hash_updates(0, total_steps, env.rank)
    metadata = {
        "schema_version": 1,
        "profile": "C6/C7 distributed backend communication trace",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "variant": variant,
        "run_id": args.profiler_run_id,
        "rank": env.rank,
        "local_rank": env.local_rank,
        "world_size": env.world_size,
        "schedule": {
            "startup_unprofiled_warmup_steps": startup_steps,
            "wait_steps": wait_steps,
            "profiler_warmup_steps": profiler_warmup_steps,
            "active_steps": active_steps,
        },
        "configuration": {
            **vars(args),
            "model_parameter_count": model_parameter_count,
            "local_micro_steps": scheduler.local_micro_steps,
            "global_tokens_per_update": (
                args.global_windows_per_update * args.block_size
            ),
            **(extra_configuration or {}),
        },
        "environment": environment_metadata(env),
        "git": git_metadata(),
        "correctness": {
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "first_observed_loss": losses[0],
            "last_observed_loss": losses[-1],
            "window_ids_sha256": window_hasher.hexdigest(),
            "expected_window_ids_sha256": expected_hash,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
            "peak_process_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "artifacts": {
            "trace": str(trace_path),
            "key_averages_table": str(table_path),
            "key_averages_json": str(events_path),
        },
        "warning": (
            "Profiler timings include instrumentation overhead and are not "
            "formal throughput results."
        ),
    }
    metadata_path = results_dir / f"{artifact_stem}_metadata.json"
    atomic_json_dump(metadata, metadata_path)
    torch.cuda.empty_cache()
    gathered = gather_rank_objects(metadata, env)

    manifest_path = None
    if env.master:
        manifest = {
            "schema_version": 1,
            "profile": "C6/C7 distributed backend communication trace",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "variant": variant,
            "run_id": args.profiler_run_id,
            "world_size": env.world_size,
            "all_losses_finite": all(
                item["correctness"]["all_losses_finite"] for item in gathered
            ),
            "all_window_hashes_match_plan": all(
                item["correctness"]["window_ids_sha256"]
                == item["correctness"]["expected_window_ids_sha256"]
                for item in gathered
            ),
            "ranks": gathered,
            "warning": (
                "Profiler timings include instrumentation overhead and must "
                "not be used as formal throughput."
            ),
        }
        manifest_path = results_dir / f"{args.profiler_run_id}_manifest.json"
        atomic_json_dump(manifest, manifest_path)
        print(f"saved profiler manifest {manifest_path}", flush=True)
    barrier()
    return manifest_path

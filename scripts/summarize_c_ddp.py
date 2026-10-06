"""Validate and summarize RouteScale C-stage strong/weak scaling runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/c_ddp/strong")
    )
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="*",
        help="required GPU counts; by default discover all *gpu directories",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_world_runs(
    results_dir: Path, world_size: int, expected_runs: int
) -> tuple[list[Path], list[dict]]:
    world_dir = results_dir / f"{world_size}gpu"
    paths = sorted(world_dir.glob("run*.json"))
    if len(paths) != expected_runs:
        raise RuntimeError(
            f"expected {expected_runs} run files in {world_dir}, found {len(paths)}"
        )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = runs[0]
    invariants = {
        "schema_version": lambda run: run["schema_version"],
        "scaling_mode": lambda run: run["scaling_mode"],
        "source_sha256": lambda run: run["version"]["source_sha256"],
        "dataset": lambda run: run["data"]["dataset"],
        "data_seed": lambda run: run["data"]["data_seed"],
        "measurement_global_window_ids_sha256": (
            lambda run: run["data"]["measurement_global_window_ids_sha256"]
        ),
        "world_size": lambda run: run["configuration"]["world_size"],
        "global_tokens_per_update": (
            lambda run: run["configuration"]["global_tokens_per_update"]
        ),
        "model_parameter_count": (
            lambda run: run["configuration"]["model_parameter_count"]
        ),
        "measured_steps": lambda run: run["measurement"]["measured_steps"],
    }
    for name, getter in invariants.items():
        expected = getter(reference)
        if any(getter(run) != expected for run in runs[1:]):
            raise RuntimeError(
                f"benchmark invariant differs across {world_size}-GPU runs: {name}"
            )
    if reference["configuration"]["world_size"] != world_size:
        raise RuntimeError(f"directory {world_dir} contains the wrong world size")
    if len({run["run_id"] for run in runs}) != len(runs):
        raise RuntimeError(f"duplicate run_id in {world_dir}")
    for run in runs:
        measurement = run["measurement"]
        if not measurement["all_losses_finite"]:
            raise RuntimeError(f"non-finite loss in {world_dir}/{run['run_id']}")
        if not measurement["all_window_hashes_match_plan"]:
            raise RuntimeError(f"window hash mismatch in {world_dir}/{run['run_id']}")
        if not all(
            math.isfinite(value)
            for value in measurement["slowest_rank_step_times_ms"]
        ):
            raise RuntimeError(f"non-finite timing in {world_dir}/{run['run_id']}")
    return paths, runs


def main() -> None:
    args = parse_args()
    if args.expected_runs < 1:
        raise ValueError("expected-runs must be positive")
    if args.world_sizes:
        world_sizes = sorted(set(args.world_sizes))
    else:
        world_sizes = sorted(
            int(path.name.removesuffix("gpu"))
            for path in args.results_dir.glob("*gpu")
            if path.is_dir() and path.name.removesuffix("gpu").isdigit()
        )
    if not world_sizes:
        raise RuntimeError(f"no GPU result directories found in {args.results_dir}")
    if 1 not in world_sizes:
        raise RuntimeError("a 1-GPU baseline is required for speedup and efficiency")

    loaded: dict[int, tuple[list[Path], list[dict]]] = {}
    for world_size in world_sizes:
        loaded[world_size] = load_world_runs(
            args.results_dir, world_size, args.expected_runs
        )

    scaling_modes = {
        run["scaling_mode"]
        for _, runs in loaded.values()
        for run in runs
    }
    if len(scaling_modes) != 1:
        raise RuntimeError("result directories mix strong and weak scaling runs")
    scaling_mode = scaling_modes.pop()
    if scaling_mode not in ("strong", "weak"):
        raise RuntimeError(f"unsupported scaling mode: {scaling_mode}")

    one_gpu_reference = loaded[1][1][0]
    if scaling_mode == "strong":
        expected_tokens = one_gpu_reference["configuration"][
            "global_tokens_per_update"
        ]
        expected_hash = one_gpu_reference["data"][
            "measurement_global_window_ids_sha256"
        ]
        for _, runs in loaded.values():
            for run in runs:
                if run["configuration"]["global_tokens_per_update"] != expected_tokens:
                    raise RuntimeError(
                        "strong-scaling global tokens/update differ across GPU counts"
                    )
                if run["data"]["measurement_global_window_ids_sha256"] != expected_hash:
                    raise RuntimeError(
                        "strong-scaling global window sequence differs across GPU counts"
                    )
    else:
        expected_local_tokens = one_gpu_reference["configuration"][
            "local_tokens_per_update"
        ]
        for world_size, (_, runs) in loaded.items():
            for run in runs:
                if run["configuration"]["local_tokens_per_update"] != expected_local_tokens:
                    raise RuntimeError(
                        "weak-scaling local tokens/update differ across GPU counts"
                    )
                expected_global = expected_local_tokens * world_size
                if run["configuration"]["global_tokens_per_update"] != expected_global:
                    raise RuntimeError(
                        "weak-scaling global tokens/update does not scale with world size"
                    )

    world_summaries = []
    for world_size in world_sizes:
        paths, runs = loaded[world_size]
        run_medians = [
            run["measurement"]["median_slowest_rank_step_ms"] for run in runs
        ]
        median_of_medians = median(run_medians)
        global_tokens = runs[0]["configuration"]["global_tokens_per_update"]
        global_throughput = global_tokens / (median_of_medians / 1000)
        world_summaries.append(
            {
                "world_size": world_size,
                "run_files": [str(path.relative_to(args.results_dir)) for path in paths],
                "run_ids": [run["run_id"] for run in runs],
                "global_tokens_per_update": global_tokens,
                "local_tokens_per_update": runs[0]["configuration"][
                    "local_tokens_per_update"
                ],
                "median_slowest_rank_step_ms_by_run": run_medians,
                "median_of_run_medians_ms": median_of_medians,
                "global_tokens_per_second": global_throughput,
                "per_gpu_tokens_per_second": global_throughput / world_size,
                "peak_allocated_mib_by_run_and_rank": [
                    [
                        rank["peak_allocated_mib"]
                        for rank in run["measurement"]["per_rank"]
                    ]
                    for run in runs
                ],
                "peak_reserved_mib_by_run_and_rank": [
                    [
                        rank["peak_reserved_mib"]
                        for rank in run["measurement"]["per_rank"]
                    ]
                    for run in runs
                ],
            }
        )

    baseline_throughput = world_summaries[0]["global_tokens_per_second"]
    for item in world_summaries:
        speedup = item["global_tokens_per_second"] / baseline_throughput
        item["speedup_vs_1gpu"] = speedup
        item["scaling_efficiency"] = speedup / item["world_size"]

    summary = {
        "schema_version": 1,
        "benchmark": "C deterministic DDP no-Profiler benchmark summary",
        "scaling_mode": scaling_mode,
        "expected_runs_per_world_size": args.expected_runs,
        "world_sizes": world_sizes,
        "world_size_results": world_summaries,
        "all_losses_finite": True,
        "all_window_hashes_match_plan": True,
    }
    output_path = args.output or (args.results_dir / "summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"C {scaling_mode} scaling summary")
    for item in world_summaries:
        print(
            f"  {item['world_size']} GPU: {item['median_of_run_medians_ms']:.3f} ms, "
            f"{item['global_tokens_per_second']:,.0f} tokens/s, "
            f"speedup {item['speedup_vs_1gpu']:.3f}x, "
            f"efficiency {item['scaling_efficiency']:.1%}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

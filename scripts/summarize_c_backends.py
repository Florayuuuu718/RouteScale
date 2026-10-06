#!/usr/bin/env python3
"""Validate and summarize formal DeepSpeed/FSDP2 C-stage benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/c_backend_benchmark"),
    )
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--variants",
        nargs="*",
        help="backend directories; by default discover every child directory",
    )
    parser.add_argument(
        "--ddp-summary",
        type=Path,
        help="optional native DDP strong-scaling summary for same-host comparison",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_runs(
    root: Path, variant: str, world_size: int, expected_runs: int
) -> list[dict]:
    directory = root / variant / f"{world_size}gpu"
    paths = sorted(directory.glob("*.json"))
    if len(paths) != expected_runs:
        raise RuntimeError(
            f"expected {expected_runs} runs in {directory}, found {len(paths)}"
        )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    invariants = {
        "variant": lambda run: run["variant"],
        "manifest": lambda run: run["data"]["manifest_sha256"],
        "measurement_hash": (
            lambda run: run["data"]["measurement_global_window_ids_sha256"]
        ),
        "world_size": lambda run: run["configuration"]["world_size"],
        "global_tokens": (
            lambda run: run["configuration"]["global_tokens_per_update"]
        ),
        "model_parameters": (
            lambda run: run["configuration"]["model_parameter_count"]
        ),
        "dtype": lambda run: run["configuration"]["dtype"],
        "warmup": lambda run: run["measurement"]["warmup_steps"],
        "measured": lambda run: run["measurement"]["measured_steps"],
    }
    reference = runs[0]
    for name, getter in invariants.items():
        expected = getter(reference)
        if any(getter(run) != expected for run in runs[1:]):
            raise RuntimeError(
                f"{variant}/{world_size}gpu invariant differs: {name}"
            )
    if reference["variant"] != variant:
        raise RuntimeError(f"{directory} contains variant {reference['variant']}")
    if reference["configuration"]["world_size"] != world_size:
        raise RuntimeError(f"{directory} contains the wrong world size")
    if len({run["run_id"] for run in runs}) != len(runs):
        raise RuntimeError(f"duplicate run IDs in {directory}")
    for run in runs:
        measurement = run["measurement"]
        if run["status"] != "benchmark_complete":
            raise RuntimeError(f"incomplete benchmark in {directory}")
        if not measurement["all_losses_finite"]:
            raise RuntimeError(f"non-finite loss in {directory}/{run['run_id']}")
        if not measurement["all_window_hashes_match_plan"]:
            raise RuntimeError(f"window hash mismatch in {directory}/{run['run_id']}")
        if not all(
            math.isfinite(value)
            for value in measurement["slowest_rank_step_times_ms"]
        ):
            raise RuntimeError(f"non-finite timing in {directory}/{run['run_id']}")
    return runs


def summarize_variant(root: Path, variant: str, worlds: list[int], runs: int) -> dict:
    loaded = {world: load_runs(root, variant, world, runs) for world in worlds}
    reference = loaded[worlds[0]][0]
    expected_tokens = reference["configuration"]["global_tokens_per_update"]
    expected_hash = reference["data"]["measurement_global_window_ids_sha256"]
    for world_runs in loaded.values():
        for run in world_runs:
            if run["configuration"]["global_tokens_per_update"] != expected_tokens:
                raise RuntimeError(f"{variant} does not keep global work fixed")
            if run["data"]["measurement_global_window_ids_sha256"] != expected_hash:
                raise RuntimeError(f"{variant} global window sequence differs by world size")

    rows = []
    for world in worlds:
        world_runs = loaded[world]
        medians = [
            run["measurement"]["median_slowest_rank_step_ms"]
            for run in world_runs
        ]
        median_ms = median(medians)
        throughput = expected_tokens / (median_ms / 1000)
        rows.append(
            {
                "world_size": world,
                "run_ids": [run["run_id"] for run in world_runs],
                "run_median_step_ms": medians,
                "median_of_run_medians_ms": median_ms,
                "global_tokens_per_second": throughput,
                "peak_allocated_bytes_by_run_and_rank": [
                    [
                        rank["peak_allocated_bytes"]
                        for rank in run["measurement"]["per_rank"]
                    ]
                    for run in world_runs
                ],
                "peak_reserved_bytes_by_run_and_rank": [
                    [
                        rank["peak_reserved_bytes"]
                        for rank in run["measurement"]["per_rank"]
                    ]
                    for run in world_runs
                ],
            }
        )
    baseline = rows[0]["global_tokens_per_second"]
    for row in rows:
        speedup = row["global_tokens_per_second"] / baseline
        row["speedup_vs_1gpu"] = speedup
        row["scaling_efficiency"] = speedup / row["world_size"]
    return {
        "variant": variant,
        "backend": reference["backend"],
        "model_parameter_count": reference["configuration"]["model_parameter_count"],
        "global_tokens_per_update": expected_tokens,
        "dtype": reference["configuration"]["dtype"],
        "fused_optimizer": reference["configuration"]["fused_optimizer"],
        "manifest_sha256": reference["data"]["manifest_sha256"],
        "measurement_global_window_ids_sha256": expected_hash,
        "world_size_results": rows,
    }


def main() -> None:
    args = parse_args()
    if args.expected_runs <= 0:
        raise ValueError("expected-runs must be positive")
    variants = args.variants or sorted(
        path.name for path in args.results_dir.iterdir() if path.is_dir()
    )
    if not variants:
        raise RuntimeError(f"no backend variants found in {args.results_dir}")
    worlds = sorted(set(args.world_sizes))
    summaries = [
        summarize_variant(args.results_dir, variant, worlds, args.expected_runs)
        for variant in variants
    ]
    comparable_fields = (
        "model_parameter_count",
        "global_tokens_per_update",
        "dtype",
        "fused_optimizer",
        "manifest_sha256",
        "measurement_global_window_ids_sha256",
    )
    reference = summaries[0]
    for summary in summaries[1:]:
        for field in comparable_fields:
            if summary[field] != reference[field]:
                raise RuntimeError(
                    f"backend invariant differs for {summary['variant']}: {field}"
                )

    ddp_rows = None
    if args.ddp_summary:
        ddp = json.loads(args.ddp_summary.read_text(encoding="utf-8"))
        ddp_rows = {
            row["world_size"]: row for row in ddp["world_size_results"]
        }
        for summary in summaries:
            for row in summary["world_size_results"]:
                ddp_row = ddp_rows.get(row["world_size"])
                if ddp_row is not None:
                    row["throughput_vs_native_ddp"] = (
                        row["global_tokens_per_second"]
                        / ddp_row["global_tokens_per_second"]
                    )

    result = {
        "schema_version": 1,
        "benchmark": "C6/C7 distributed backend benchmark summary",
        "expected_runs_per_world_size": args.expected_runs,
        "world_sizes": worlds,
        "variants": summaries,
        "native_ddp_summary": str(args.ddp_summary) if args.ddp_summary else None,
        "all_losses_finite": True,
        "all_window_hashes_match_plan": True,
    }
    output = args.output or (args.results_dir / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("C6/C7 backend benchmark summary")
    for summary in summaries:
        print(f"  {summary['variant']}")
        for row in summary["world_size_results"]:
            comparison = row.get("throughput_vs_native_ddp")
            suffix = (
                f", {comparison:.3f}x native DDP"
                if comparison is not None
                else ""
            )
            print(
                f"    {row['world_size']} GPU: "
                f"{row['median_of_run_medians_ms']:.3f} ms, "
                f"{row['global_tokens_per_second']:,.0f} tokens/s, "
                f"speedup {row['speedup_vs_1gpu']:.3f}x, "
                f"efficiency {row['scaling_efficiency']:.1%}{suffix}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

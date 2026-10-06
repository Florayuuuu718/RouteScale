"""Validate and summarize independent RouteScale B1 benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/b1_single_gpu")
    )
    parser.add_argument("--expected-runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_paths = sorted(args.results_dir.glob("run*.json"))
    if len(run_paths) != args.expected_runs:
        raise RuntimeError(
            f"expected {args.expected_runs} run files, found {len(run_paths)}"
        )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    reference = runs[0]
    invariant_paths = {
        "git_commit": lambda run: run["version"]["git_commit"],
        "source_sha256": lambda run: run["version"]["source_sha256"],
        "dataset": lambda run: run["data"]["dataset"],
        "data_seed": lambda run: run["data"]["data_seed"],
        "measurement_offsets_sha256": (
            lambda run: run["data"]["measurement_offsets_sha256"]
        ),
        "tokens_per_update": (
            lambda run: run["configuration"]["tokens_per_update"]
        ),
        "model_parameter_count": (
            lambda run: run["configuration"]["model_parameter_count"]
        ),
    }
    for name, getter in invariant_paths.items():
        expected = getter(reference)
        if any(getter(run) != expected for run in runs[1:]):
            raise RuntimeError(f"benchmark invariant differs across runs: {name}")
    if not all(run["measurement"]["all_losses_finite"] for run in runs):
        raise RuntimeError("a benchmark run contains NaN or Inf loss")

    run_medians = [run["measurement"]["median_step_ms"] for run in runs]
    median_of_medians = median(run_medians)
    tokens_per_update = reference["configuration"]["tokens_per_update"]
    summary = {
        "schema_version": 1,
        "benchmark": "B1 single-GPU no-Profiler baseline summary",
        "run_files": [path.name for path in run_paths],
        "run_ids": [run["run_id"] for run in runs],
        "measurement_offsets_sha256": reference["data"][
            "measurement_offsets_sha256"
        ],
        "tokens_per_update": tokens_per_update,
        "measured_steps_per_run": reference["measurement"]["measured_steps"],
        "median_step_ms_by_run": run_medians,
        "median_of_run_medians_ms": median_of_medians,
        "tokens_per_second_from_median_of_medians": (
            tokens_per_update / (median_of_medians / 1000)
        ),
        "peak_allocated_mib_by_run": [
            run["measurement"]["peak_allocated_mib"] for run in runs
        ],
        "peak_reserved_mib_by_run": [
            run["measurement"]["peak_reserved_mib"] for run in runs
        ],
        "all_losses_finite": True,
        "all_values_finite": all(
            math.isfinite(value)
            for run in runs
            for value in run["measurement"]["step_times_ms"]
        ),
    }
    output_path = args.results_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"B1 summary: {median_of_medians:.3f} ms/update, "
        f"{summary['tokens_per_second_from_median_of_medians']:,.0f} tokens/s"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

"""Summarize baseline and compiled PyTorch Profiler artifacts for B2."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/b2_profiler")
    )
    return parser.parse_args()


def parse_time_total(table_text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s+([0-9.]+)(ms|s|us)", table_text)
    if match is None:
        raise RuntimeError(f"missing {label} in key-averages table")
    value = float(match.group(1))
    unit = match.group(2)
    return value * {"us": 0.001, "ms": 1.0, "s": 1000.0}[unit]


def read_profile(results_dir: Path, name: str) -> dict:
    rows = json.loads(
        (results_dir / f"{name}_key_averages.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (results_dir / f"{name}_metadata.json").read_text(encoding="utf-8")
    )
    table = (results_dir / f"{name}_key_averages.txt").read_text(encoding="utf-8")
    by_name = {row["name"]: row for row in rows}

    def event(event_name: str) -> dict:
        if event_name not in by_name:
            raise RuntimeError(f"{name} profile is missing event {event_name}")
        return by_name[event_name]

    cuda_launch = event("cudaLaunchKernel")["count"]
    cu_launch = event("cuLaunchKernel")["count"]
    return {
        "run_id": name,
        "compile": metadata["configuration"]["compile"],
        "active_steps": metadata["schedule"]["active_steps"],
        "all_losses_finite": metadata["correctness"]["all_losses_finite"],
        "self_cpu_time_total_ms": parse_time_total(table, "Self CPU time total"),
        "self_cuda_time_total_ms": parse_time_total(table, "Self CUDA time total"),
        "aten_mm": event("aten::mm"),
        "aten_copy": event("aten::copy_"),
        "h2d": event("Memcpy HtoD (Pinned -> Device)"),
        "gradient_clipping": event("gradient_clipping"),
        "optimizer_step": event("optimizer_step"),
        "cuda_launch_kernel_count": cuda_launch,
        "cu_launch_kernel_count": cu_launch,
        "combined_launch_api_count": cuda_launch + cu_launch,
    }


def percent_reduction(before: float, after: float) -> float:
    return (1 - after / before) * 100


def main() -> None:
    args = parse_args()
    baseline = read_profile(args.results_dir, "baseline")
    compiled = read_profile(args.results_dir, "compiled")
    if baseline["compile"] or not compiled["compile"]:
        raise RuntimeError("expected compile=False baseline and compile=True comparison")
    if baseline["active_steps"] != compiled["active_steps"]:
        raise RuntimeError("profile active-step count differs")
    if not baseline["all_losses_finite"] or not compiled["all_losses_finite"]:
        raise RuntimeError("a profile contains NaN or Inf loss")

    summary = {
        "schema_version": 1,
        "experiment": "B2 PyTorch Profiler eager/compiled trace comparison",
        "warning": (
            "Profiler totals include instrumentation overhead and overlapping "
            "abstraction levels; formal speed comes from the no-Profiler A/B."
        ),
        "baseline": baseline,
        "compiled": compiled,
        "comparison": {
            "self_cpu_time_reduction_percent": percent_reduction(
                baseline["self_cpu_time_total_ms"],
                compiled["self_cpu_time_total_ms"],
            ),
            "self_cuda_time_reduction_percent": percent_reduction(
                baseline["self_cuda_time_total_ms"],
                compiled["self_cuda_time_total_ms"],
            ),
            "combined_launch_api_count_reduction_percent": percent_reduction(
                baseline["combined_launch_api_count"],
                compiled["combined_launch_api_count"],
            ),
            "aten_copy_call_reduction_percent": percent_reduction(
                baseline["aten_copy"]["count"], compiled["aten_copy"]["count"]
            ),
            "aten_mm_self_device_time_reduction_percent": percent_reduction(
                baseline["aten_mm"]["self_device_time_us"],
                compiled["aten_mm"]["self_device_time_us"],
            ),
        },
    }
    output_path = args.results_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = summary["comparison"]
    print(
        "B2 profiler summary: "
        f"CUDA self time {comparison['self_cuda_time_reduction_percent']:.1f}% lower, "
        f"launch API calls {comparison['combined_launch_api_count_reduction_percent']:.1f}% lower, "
        f"aten::copy_ calls {comparison['aten_copy_call_reduction_percent']:.1f}% lower"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

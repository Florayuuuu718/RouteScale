"""Validate the B2 eager/torch.compile no-Profiler A/B benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eager-dir", type=Path, default=Path("results/b2_compile_ab/eager")
    )
    parser.add_argument(
        "--compiled-dir", type=Path, default=Path("results/b2_compile_ab/compiled")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/b2_compile_ab/summary.json")
    )
    return parser.parse_args()


def load_group(path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    runs = [
        json.loads(run_path.read_text(encoding="utf-8"))
        for run_path in sorted(path.glob("run*.json"))
    ]
    if len(runs) != 3:
        raise RuntimeError(f"expected three runs in {path}, found {len(runs)}")
    return summary, runs


def main() -> None:
    args = parse_args()
    eager_summary, eager_runs = load_group(args.eager_dir)
    compiled_summary, compiled_runs = load_group(args.compiled_dir)
    all_runs = eager_runs + compiled_runs

    reference = eager_runs[0]
    invariants = {
        "measurement_offsets_sha256": (
            lambda run: run["data"]["measurement_offsets_sha256"]
        ),
        "tokens_per_update": (
            lambda run: run["configuration"]["tokens_per_update"]
        ),
        "model_parameter_count": (
            lambda run: run["configuration"]["model_parameter_count"]
        ),
        "batch_size": lambda run: run["configuration"]["batch_size"],
        "block_size": lambda run: run["configuration"]["block_size"],
        "gradient_accumulation_steps": (
            lambda run: run["configuration"]["effective_gradient_accumulation_steps"]
        ),
        "n_layer": lambda run: run["configuration"]["n_layer"],
        "n_head": lambda run: run["configuration"]["n_head"],
        "n_embd": lambda run: run["configuration"]["n_embd"],
        "dtype": lambda run: run["configuration"]["dtype"],
        "warmup_steps": lambda run: run["measurement"]["warmup_steps"],
        "measured_steps": lambda run: run["measurement"]["measured_steps"],
    }
    for name, getter in invariants.items():
        expected = getter(reference)
        if any(getter(run) != expected for run in all_runs[1:]):
            raise RuntimeError(f"A/B invariant differs: {name}")
    if any(run["configuration"]["compile"] for run in eager_runs):
        raise RuntimeError("eager group unexpectedly has compile=True")
    if not all(run["configuration"]["compile"] for run in compiled_runs):
        raise RuntimeError("compiled group does not consistently have compile=True")
    if not all(run["measurement"]["all_losses_finite"] for run in all_runs):
        raise RuntimeError("an A/B run contains NaN or Inf loss")

    eager_ms = eager_summary["median_of_run_medians_ms"]
    compiled_ms = compiled_summary["median_of_run_medians_ms"]
    eager_throughput = eager_summary["tokens_per_second_from_median_of_medians"]
    compiled_throughput = compiled_summary[
        "tokens_per_second_from_median_of_medians"
    ]
    result = {
        "schema_version": 1,
        "experiment": "B2 no-Profiler torch.compile A/B",
        "changed_variable": {"compile": {"before": False, "after": True}},
        "invariants": {
            name: getter(reference) for name, getter in invariants.items()
        },
        "eager": {
            "median_step_ms_by_run": eager_summary["median_step_ms_by_run"],
            "median_of_run_medians_ms": eager_ms,
            "tokens_per_second": eager_throughput,
            "median_peak_allocated_mib": median(
                eager_summary["peak_allocated_mib_by_run"]
            ),
            "median_peak_reserved_mib": median(
                eager_summary["peak_reserved_mib_by_run"]
            ),
        },
        "compiled": {
            "median_step_ms_by_run": compiled_summary["median_step_ms_by_run"],
            "median_of_run_medians_ms": compiled_ms,
            "tokens_per_second": compiled_throughput,
            "median_peak_allocated_mib": median(
                compiled_summary["peak_allocated_mib_by_run"]
            ),
            "median_peak_reserved_mib": median(
                compiled_summary["peak_reserved_mib_by_run"]
            ),
        },
        "comparison": {
            "speedup": eager_ms / compiled_ms,
            "step_time_reduction_percent": (1 - compiled_ms / eager_ms) * 100,
            "throughput_increase_percent": (
                compiled_throughput / eager_throughput - 1
            ) * 100,
            "peak_allocated_change_mib": median(
                compiled_summary["peak_allocated_mib_by_run"]
            ) - median(eager_summary["peak_allocated_mib_by_run"]),
            "peak_reserved_change_mib": median(
                compiled_summary["peak_reserved_mib_by_run"]
            ) - median(eager_summary["peak_reserved_mib_by_run"]),
            "all_losses_finite": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"B2 A/B: {eager_ms:.3f} -> {compiled_ms:.3f} ms/update, "
        f"{result['comparison']['speedup']:.3f}x speedup, "
        f"{result['comparison']['throughput_increase_percent']:.1f}% throughput"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

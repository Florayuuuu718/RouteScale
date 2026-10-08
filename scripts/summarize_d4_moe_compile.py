"""Validate and summarize D4 eager/torch.compile Dense and MoE benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


VARIANTS = ("dense", "moe")
CORE_CONFIGURATION_FIELDS = (
    "dataset",
    "batch_size",
    "block_size",
    "effective_gradient_accumulation_steps",
    "n_layer",
    "n_head",
    "n_embd",
    "dtype",
    "benchmark_warmup_steps",
    "benchmark_measure_steps",
    "benchmark_data_seed",
    "tokens_per_update",
    "model_parameter_count",
    "active_parameter_count",
    "moe_num_experts",
    "moe_layer_index",
    "moe_capacity_factor",
    "moe_drop_tokens",
    "moe_balance_loss_weight",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(root: Path, variant: str) -> list[dict[str, Any]]:
    paths = [root / variant / f"run{index}.json" for index in (1, 2, 3)]
    runs = [read_json(path) for path in paths]
    if [run["run_id"] for run in runs] != ["run1", "run2", "run3"]:
        raise ValueError(f"{root / variant} does not contain run1/run2/run3")
    if not all(run["measurement"]["all_losses_finite"] for run in runs):
        raise ValueError(f"{root / variant} contains a non-finite loss")
    return runs


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [run["measurement"]["median_step_ms"] for run in runs]
    middle = median(medians)
    tokens_per_update = runs[0]["configuration"]["tokens_per_update"]
    return {
        "run_median_step_ms": medians,
        "median_of_run_medians_ms": middle,
        "tokens_per_second": tokens_per_update / (middle / 1000),
        "run_median_range_ms": max(medians) - min(medians),
        "median_peak_allocated_mib": median(
            run["measurement"]["peak_allocated_mib"] for run in runs
        ),
        "median_peak_reserved_mib": median(
            run["measurement"]["peak_reserved_mib"] for run in runs
        ),
    }


def assert_same(values: list[Any], message: str) -> None:
    if any(value != values[0] for value in values[1:]):
        raise ValueError(message)


def validate_pair(
    eager_runs: list[dict[str, Any]], compiled_runs: list[dict[str, Any]], variant: str
) -> dict[str, Any]:
    all_runs = eager_runs + compiled_runs
    if any(run["configuration"]["compile"] for run in eager_runs):
        raise ValueError(f"{variant} eager runs unexpectedly use compile=True")
    if not all(run["configuration"]["compile"] for run in compiled_runs):
        raise ValueError(f"{variant} compiled runs do not all use compile=True")

    for field in CORE_CONFIGURATION_FIELDS:
        assert_same(
            [run["configuration"][field] for run in all_runs],
            f"{variant} eager/compiled invariant differs: {field}",
        )
    for field in ("warmup_offsets_sha256", "measurement_offsets_sha256"):
        assert_same(
            [run["data"][field] for run in all_runs],
            f"{variant} eager/compiled data differs: {field}",
        )
    for path in ("train.py", "model.py"):
        assert_same(
            [run["version"]["source_sha256"][path] for run in all_runs],
            f"{variant} eager/compiled source differs: {path}",
        )
    if variant == "moe":
        assert_same(
            [run["version"]["source_sha256"]["moe.py"] for run in all_runs],
            "MoE eager/compiled source differs: moe.py",
        )

    max_loss_difference = 0.0
    for eager, compiled in zip(eager_runs, compiled_runs, strict=True):
        eager_losses = eager["measurement"]["last_micro_batch_lm_losses"]
        compiled_losses = compiled["measurement"]["last_micro_batch_lm_losses"]
        if len(eager_losses) != len(compiled_losses):
            raise ValueError(f"{variant} eager/compiled loss lengths differ")
        max_loss_difference = max(
            max_loss_difference,
            max(abs(left - right) for left, right in zip(eager_losses, compiled_losses)),
        )
    if not math.isfinite(max_loss_difference):
        raise ValueError(f"{variant} loss comparison is non-finite")

    return {
        "same_configuration_except_compile_and_artifact_paths": True,
        "same_warmup_windows": True,
        "same_measurement_windows": True,
        "same_training_sources": True,
        "max_absolute_measured_lm_loss_difference": max_loss_difference,
    }


def compare(eager: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
    eager_ms = eager["median_of_run_medians_ms"]
    compiled_ms = compiled["median_of_run_medians_ms"]
    return {
        "speedup": eager_ms / compiled_ms,
        "step_time_change_percent": (compiled_ms / eager_ms - 1.0) * 100,
        "throughput_change_percent": (
            compiled["tokens_per_second"] / eager["tokens_per_second"] - 1.0
        ) * 100,
        "peak_allocated_change_mib": (
            compiled["median_peak_allocated_mib"]
            - eager["median_peak_allocated_mib"]
        ),
        "peak_reserved_change_mib": (
            compiled["median_peak_reserved_mib"]
            - eager["median_peak_reserved_mib"]
        ),
    }


def summarize(
    eager_root: Path, compiled_root: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "D4 Dense/MoE eager versus torch.compile",
        "changed_variable": {"compile": {"eager": False, "compiled": True}},
        "variants": {},
    }
    groups: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        eager_runs = load_runs(eager_root, variant)
        compiled_runs = load_runs(compiled_root, variant)
        invariants = validate_pair(eager_runs, compiled_runs, variant)
        eager_summary = summarize_runs(eager_runs)
        compiled_summary = summarize_runs(compiled_runs)
        groups[variant] = {"eager": eager_summary, "compiled": compiled_summary}
        result["variants"][variant] = {
            "invariants": invariants,
            "eager": eager_summary,
            "compiled": compiled_summary,
            "comparison": compare(eager_summary, compiled_summary),
        }

    dense_compiled = groups["dense"]["compiled"]
    moe_compiled = groups["moe"]["compiled"]
    result["compiled_moe_vs_dense"] = {
        "throughput_ratio": (
            moe_compiled["tokens_per_second"]
            / dense_compiled["tokens_per_second"]
        ),
        "step_time_change_percent": (
            moe_compiled["median_of_run_medians_ms"]
            / dense_compiled["median_of_run_medians_ms"]
            - 1.0
        ) * 100,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eager-root", type=Path, default=Path("results/d3_moe/benchmark")
    )
    parser.add_argument(
        "--compiled-root",
        type=Path,
        default=Path("results/d4_moe_compile/benchmark"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/d4_moe_compile/summary.json"),
    )
    args = parser.parse_args()
    result = summarize(args.eager_root, args.compiled_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for variant in VARIANTS:
        row = result["variants"][variant]
        print(
            f"{variant}: {row['eager']['median_of_run_medians_ms']:.3f} -> "
            f"{row['compiled']['median_of_run_medians_ms']:.3f} ms/update, "
            f"{row['comparison']['speedup']:.3f}x"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Validate and summarize the E-stage dispatch/compile benchmark matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


VARIANTS = (
    "dense_eager",
    "dense_compiled",
    "loop_eager",
    "loop_compiled",
    "padded_torch_eager",
    "padded_torch_compiled",
    "padded_triton_eager",
    "padded_triton_compiled",
)

FROZEN_CONFIGURATION_KEYS = (
    "dataset",
    "gradient_accumulation_steps",
    "batch_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
    "dtype",
    "benchmark_warmup_steps",
    "benchmark_measure_steps",
    "benchmark_data_seed",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(root: Path, variant: str) -> list[dict[str, Any]]:
    paths = [
        root / "benchmark" / variant / f"run{index}.json"
        for index in (1, 2, 3)
    ]
    runs = [read_json(path) for path in paths]
    if [run["run_id"] for run in runs] != ["run1", "run2", "run3"]:
        raise ValueError(f"{variant} run IDs are incomplete")
    if not all(
        run["measurement"]["all_losses_finite"] for run in runs
    ):
        raise ValueError(f"{variant} contains a non-finite loss")
    return runs


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [
        run["measurement"]["median_step_ms"] for run in runs
    ]
    selected_median = median(medians)
    tokens_per_update = runs[0]["configuration"]["tokens_per_update"]
    allocated = [
        run["measurement"]["peak_allocated_mib"] for run in runs
    ]
    reserved = [
        run["measurement"]["peak_reserved_mib"] for run in runs
    ]
    return {
        "run_median_step_ms": medians,
        "median_of_run_medians_ms": selected_median,
        "run_median_range_ms": max(medians) - min(medians),
        "tokens_per_second": tokens_per_update / (selected_median / 1000),
        "median_peak_allocated_mib": median(allocated),
        "median_peak_reserved_mib": median(reserved),
        "model_parameter_count": runs[0]["configuration"][
            "model_parameter_count"
        ],
        "active_parameter_count": runs[0]["configuration"][
            "active_parameter_count"
        ],
        "compile": runs[0]["configuration"]["compile"],
        "dispatch_backend": runs[0]["configuration"][
            "moe_dispatch_backend"
        ],
    }


def compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict:
    candidate_step = candidate["median_of_run_medians_ms"]
    baseline_step = baseline["median_of_run_medians_ms"]
    candidate_throughput = candidate["tokens_per_second"]
    baseline_throughput = baseline["tokens_per_second"]
    return {
        "speedup": baseline_step / candidate_step,
        "step_time_change_percent": (
            candidate_step / baseline_step - 1.0
        )
        * 100,
        "throughput_change_percent": (
            candidate_throughput / baseline_throughput - 1.0
        )
        * 100,
        "peak_allocated_change_mib": (
            candidate["median_peak_allocated_mib"]
            - baseline["median_peak_allocated_mib"]
        ),
        "peak_reserved_change_mib": (
            candidate["median_peak_reserved_mib"]
            - baseline["median_peak_reserved_mib"]
        ),
    }


def validate_invariants(
    runs_by_variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    every_run = [
        run
        for runs in runs_by_variant.values()
        for run in runs
    ]
    reference = every_run[0]
    data_keys = (
        "warmup_offsets_sha256",
        "measurement_offsets_sha256",
        "warmup_window_count",
        "measurement_window_count",
    )
    for run in every_run[1:]:
        for key in FROZEN_CONFIGURATION_KEYS:
            if run["configuration"][key] != reference["configuration"][key]:
                raise ValueError(f"configuration mismatch for {key}")
        for key in data_keys:
            if run["data"][key] != reference["data"][key]:
                raise ValueError(f"data plan mismatch for {key}")

    moe_runs = [
        run
        for variant, runs in runs_by_variant.items()
        if not variant.startswith("dense")
        for run in runs
    ]
    expected_moe = {
        "moe_num_experts": 4,
        "moe_layer_index": 4,
        "moe_capacity_factor": 1.25,
        "moe_drop_tokens": True,
        "moe_balance_loss_weight": 0.01,
    }
    for run in moe_runs:
        for key, expected in expected_moe.items():
            if run["configuration"][key] != expected:
                raise ValueError(f"MoE invariant mismatch for {key}")

    source_paths = ("train.py", "model.py", "config/benchmark_e_moe.py")
    for path in source_paths:
        hashes = {
            run["version"]["source_sha256"][path] for run in every_run
        }
        if len(hashes) != 1:
            raise ValueError(f"source hash mismatch for {path}")
    for path in ("moe.py", "triton_grouped_gemm.py"):
        hashes = {
            run["version"]["source_sha256"][path] for run in moe_runs
        }
        if len(hashes) != 1:
            raise ValueError(f"MoE source hash mismatch for {path}")
    return {
        "same_frozen_configuration": True,
        "same_data_windows": True,
        "same_common_source": True,
        "same_moe_source": True,
        "warmup_offsets_sha256": reference["data"][
            "warmup_offsets_sha256"
        ],
        "measurement_offsets_sha256": reference["data"][
            "measurement_offsets_sha256"
        ],
        "independent_runs_per_variant": 3,
    }


def build_summary(root: Path) -> dict[str, Any]:
    runs_by_variant = {
        variant: load_runs(root, variant) for variant in VARIANTS
    }
    variants = {
        variant: summarize_runs(runs)
        for variant, runs in runs_by_variant.items()
    }
    comparisons = {
        "padded_torch_eager_vs_loop_eager": compare(
            variants["padded_torch_eager"], variants["loop_eager"]
        ),
        "padded_triton_eager_vs_loop_eager": compare(
            variants["padded_triton_eager"], variants["loop_eager"]
        ),
        "padded_triton_eager_vs_padded_torch_eager": compare(
            variants["padded_triton_eager"],
            variants["padded_torch_eager"],
        ),
        "loop_compiled_vs_loop_eager": compare(
            variants["loop_compiled"], variants["loop_eager"]
        ),
        "padded_torch_compiled_vs_padded_torch_eager": compare(
            variants["padded_torch_compiled"],
            variants["padded_torch_eager"],
        ),
        "padded_triton_compiled_vs_padded_triton_eager": compare(
            variants["padded_triton_compiled"],
            variants["padded_triton_eager"],
        ),
        "padded_torch_compiled_vs_loop_compiled": compare(
            variants["padded_torch_compiled"],
            variants["loop_compiled"],
        ),
        "padded_triton_compiled_vs_loop_compiled": compare(
            variants["padded_triton_compiled"],
            variants["loop_compiled"],
        ),
        "padded_triton_compiled_vs_padded_torch_compiled": compare(
            variants["padded_triton_compiled"],
            variants["padded_torch_compiled"],
        ),
        "padded_triton_compiled_vs_dense_compiled": compare(
            variants["padded_triton_compiled"],
            variants["dense_compiled"],
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "E-stage fixed-capacity Triton MoE comparison",
        "invariants": validate_invariants(runs_by_variant),
        "microbenchmark": read_json(root / "grouped_gemm.json"),
        "variants": variants,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/e_triton_moe")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e_triton_moe/summary.json"),
    )
    args = parser.parse_args()
    summary = build_summary(args.root)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["comparisons"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

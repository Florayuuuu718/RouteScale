from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_runs(root: Path, variant: str) -> list[dict[str, Any]]:
    paths = [root / "benchmark" / variant / f"run{index}.json" for index in (1, 2, 3)]
    runs = [read_json(path) for path in paths]
    if [run["run_id"] for run in runs] != ["run1", "run2", "run3"]:
        raise ValueError(f"{variant} benchmark run IDs are incomplete")
    if not all(run["measurement"]["all_losses_finite"] for run in runs):
        raise ValueError(f"{variant} benchmark contains non-finite loss")
    return runs


def distribution_metrics(counts: list[int]) -> dict[str, float]:
    mean_load = fmean(counts)
    total = sum(counts)
    fractions = [count / total for count in counts if count]
    entropy = -sum(fraction * math.log(fraction) for fraction in fractions)
    return {
        "max_to_mean_load": max(counts) / mean_load,
        "load_coefficient_of_variation": pstdev(counts) / mean_load,
        "normalized_load_entropy": entropy / math.log(len(counts)),
    }


def summarize_benchmark(runs: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [run["measurement"]["median_step_ms"] for run in runs]
    run_throughputs = [
        run["measurement"]["tokens_per_second_from_median"] for run in runs
    ]
    median_step_ms = median(medians)
    tokens_per_update = runs[0]["configuration"]["tokens_per_update"]
    configuration = runs[0]["configuration"]
    return {
        "run_median_step_ms": medians,
        "run_tokens_per_second": run_throughputs,
        "median_of_run_medians_ms": median_step_ms,
        "tokens_per_second": tokens_per_update / (median_step_ms / 1000),
        "run_median_range_ms": max(medians) - min(medians),
        "median_peak_allocated_mib": median(
            run["measurement"]["peak_allocated_mib"] for run in runs
        ),
        "median_peak_reserved_mib": median(
            run["measurement"]["peak_reserved_mib"] for run in runs
        ),
        "model_parameter_count": configuration["model_parameter_count"],
        "active_parameter_count": configuration["active_parameter_count"],
        "non_position_embedding_parameter_count": configuration[
            "non_position_embedding_parameter_count"
        ],
        "active_non_position_embedding_parameter_count": configuration[
            "active_non_position_embedding_parameter_count"
        ],
        "all_losses_finite": True,
    }


def aggregate_routing(updates: list[dict[str, Any]]) -> dict[str, Any]:
    expert_count = len(updates[0]["expert_counts"])
    counts = [
        sum(update["expert_counts"][index] for update in updates)
        for index in range(expert_count)
    ]
    total_tokens = sum(update["token_count"] for update in updates)
    total_dropped = sum(update["dropped_token_count"] for update in updates)
    return {
        "update_count": len(updates),
        "expert_counts": counts,
        **distribution_metrics(counts),
        "drop_rate": total_dropped / total_tokens,
        "max_update_drop_rate": max(update["drop_rate"] for update in updates),
        "mean_update_drop_rate": fmean(update["drop_rate"] for update in updates),
        "last_update_expert_counts": updates[-1]["expert_counts"],
        "last_update_drop_rate": updates[-1]["drop_rate"],
    }


def summarize_stability(
    dense: dict[str, Any], moe: dict[str, Any]
) -> dict[str, Any]:
    dense_data = dense["data"]
    moe_data = moe["data"]
    invariant = (
        dense_data["dataset"],
        dense_data["seed"],
        dense_data["window_count"],
        dense_data["window_offsets_sha256"],
    )
    if invariant != (
        moe_data["dataset"],
        moe_data["seed"],
        moe_data["window_count"],
        moe_data["window_offsets_sha256"],
    ):
        raise ValueError("Dense and MoE stability runs used different windows")
    dense_evaluations = dense["evaluations"]
    moe_evaluations = moe["evaluations"]
    if [row["update"] for row in dense_evaluations] != [
        row["update"] for row in moe_evaluations
    ]:
        raise ValueError("Dense and MoE validation schedules differ")
    if not all(
        math.isfinite(update["mean_lm_loss"])
        for run in (dense, moe)
        for update in run["updates"]
    ):
        raise ValueError("stability run contains non-finite LM loss")

    validation = [
        {
            "update": dense_row["update"],
            "dense_val_lm_loss": dense_row["val_lm"],
            "moe_val_lm_loss": moe_row["val_lm"],
            "moe_minus_dense": moe_row["val_lm"] - dense_row["val_lm"],
        }
        for dense_row, moe_row in zip(
            dense_evaluations, moe_evaluations, strict=True
        )
    ]
    return {
        "invariants": {
            "same_data_windows": True,
            "dataset": invariant[0],
            "data_seed": invariant[1],
            "window_count": invariant[2],
            "window_offsets_sha256": invariant[3],
            "all_training_lm_losses_finite": True,
        },
        "validation_lm_loss": validation,
        "dense_parameter_count": dense["model_parameter_count"],
        "moe_parameter_count": moe["model_parameter_count"],
        "moe_routing_all_updates": aggregate_routing(moe["updates"]),
        "moe_routing_last_100_updates": aggregate_routing(moe["updates"][-100:]),
    }


def summarize_directory(root: Path) -> dict[str, Any]:
    dense_runs = load_benchmark_runs(root, "dense")
    moe_runs = load_benchmark_runs(root, "moe")
    all_runs = dense_runs + moe_runs
    measurement_hashes = {
        run["data"]["measurement_offsets_sha256"] for run in all_runs
    }
    warmup_hashes = {run["data"]["warmup_offsets_sha256"] for run in all_runs}
    invariant_fields = (
        "dataset",
        "batch_size",
        "block_size",
        "gradient_accumulation_steps",
        "n_layer",
        "n_head",
        "n_embd",
        "dtype",
        "compile",
        "benchmark_warmup_steps",
        "benchmark_measure_steps",
        "benchmark_data_seed",
    )
    invariant_values = {
        tuple(run["configuration"][field] for field in invariant_fields)
        for run in all_runs
    }
    environments = {
        json.dumps(run["environment"], sort_keys=True) for run in all_runs
    }
    common_source_hashes = {
        (
            run["version"]["source_sha256"]["train.py"],
            run["version"]["source_sha256"]["model.py"],
        )
        for run in all_runs
    }
    if not (
        len(measurement_hashes)
        == len(warmup_hashes)
        == len(invariant_values)
        == len(environments)
        == len(common_source_hashes)
        == 1
    ):
        raise ValueError("formal benchmark invariants do not match")

    dense_summary = summarize_benchmark(dense_runs)
    moe_summary = summarize_benchmark(moe_runs)
    dense_step = dense_summary["median_of_run_medians_ms"]
    moe_step = moe_summary["median_of_run_medians_ms"]
    dense_throughput = dense_summary["tokens_per_second"]
    moe_throughput = moe_summary["tokens_per_second"]
    stability = summarize_stability(
        read_json(root / "stability_dense.json"),
        read_json(root / "stability_moe.json"),
    )
    return {
        "schema_version": 1,
        "experiment": "D3 formal Dense versus Top-1 MoE comparison",
        "benchmark_invariants": {
            "same_warmup_windows": True,
            "same_measurement_windows": True,
            "warmup_offsets_sha256": next(iter(warmup_hashes)),
            "measurement_offsets_sha256": next(iter(measurement_hashes)),
            "same_workload_precision_hardware": True,
            "same_common_source": True,
            "independent_runs_per_variant": 3,
        },
        "benchmark": {
            "dense": dense_summary,
            "moe": moe_summary,
            "comparison": {
                "moe_to_dense_throughput_ratio": moe_throughput / dense_throughput,
                "moe_throughput_change_percent": (
                    moe_throughput / dense_throughput - 1.0
                ) * 100,
                "moe_step_time_change_percent": (moe_step / dense_step - 1.0) * 100,
                "peak_allocated_change_mib": (
                    moe_summary["median_peak_allocated_mib"]
                    - dense_summary["median_peak_allocated_mib"]
                ),
                "peak_reserved_change_mib": (
                    moe_summary["median_peak_reserved_mib"]
                    - dense_summary["median_peak_reserved_mib"]
                ),
                "total_parameter_change": (
                    moe_summary["model_parameter_count"]
                    - dense_summary["model_parameter_count"]
                ),
                "active_parameter_change": (
                    moe_summary["active_parameter_count"]
                    - dense_summary["active_parameter_count"]
                ),
            },
        },
        "stability": stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/d3_moe"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize_directory(args.results_dir)
    output = args.output or args.results_dir / "summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

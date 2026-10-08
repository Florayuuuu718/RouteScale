from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


VARIANTS = {
    "unbounded": {
        "moe_capacity_factor": 0.0,
        "moe_drop_tokens": False,
        "moe_balance_loss_weight": 0.0,
    },
    "capacity_observe": {
        "moe_capacity_factor": 1.25,
        "moe_drop_tokens": False,
        "moe_balance_loss_weight": 0.0,
    },
    "capacity_drop": {
        "moe_capacity_factor": 1.25,
        "moe_drop_tokens": True,
        "moe_balance_loss_weight": 0.0,
    },
    "balance_loss": {
        "moe_capacity_factor": 1.25,
        "moe_drop_tokens": True,
        "moe_balance_loss_weight": 0.01,
    },
}


def load_run(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError(f"{path} is not complete")
    if not run.get("updates"):
        raise ValueError(f"{path} has no update metrics")
    for key, value in expected.items():
        actual = run["configuration"].get(key)
        if actual != value:
            raise ValueError(f"{path}: {key}={actual!r}, expected {value!r}")
    return run


def distribution_metrics(counts: list[int]) -> dict[str, float]:
    total = sum(counts)
    mean_load = total / len(counts)
    variance = sum((count - mean_load) ** 2 for count in counts) / len(counts)
    coefficient_of_variation = (
        math.sqrt(variance) / mean_load if mean_load else 0.0
    )
    if total == 0 or len(counts) == 1:
        normalized_entropy = 1.0
    else:
        fractions = [count / total for count in counts if count]
        normalized_entropy = -sum(
            fraction * math.log(fraction) for fraction in fractions
        ) / math.log(len(counts))
    return {
        "max_to_mean_load": max(counts) / mean_load if mean_load else 0.0,
        "load_coefficient_of_variation": coefficient_of_variation,
        "normalized_load_entropy": normalized_entropy,
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    updates = run["updates"]
    expert_count = len(updates[0]["expert_counts"])
    aggregate_counts = [
        sum(update["expert_counts"][index] for update in updates)
        for index in range(expert_count)
    ]
    aggregate_processed = [
        sum(update["processed_counts"][index] for update in updates)
        for index in range(expert_count)
    ]
    total_tokens = sum(update["token_count"] for update in updates)
    total_overflow = sum(update["overflow_token_count"] for update in updates)
    total_dropped = sum(update["dropped_token_count"] for update in updates)
    return {
        "update_count": len(updates),
        "total_tokens": total_tokens,
        "aggregate_expert_counts": aggregate_counts,
        "aggregate_processed_counts": aggregate_processed,
        **distribution_metrics(aggregate_counts),
        "mean_update_max_to_mean_load": fmean(
            update["max_to_mean_load"] for update in updates
        ),
        "mean_update_load_coefficient_of_variation": fmean(
            update["load_coefficient_of_variation"] for update in updates
        ),
        "total_overflow_tokens": total_overflow,
        "overflow_rate": total_overflow / total_tokens,
        "total_dropped_tokens": total_dropped,
        "drop_rate": total_dropped / total_tokens,
        "mean_lm_loss": fmean(update["mean_lm_loss"] for update in updates),
        "mean_balance_loss": fmean(
            update["mean_balance_loss"] for update in updates
        ),
        "mean_total_loss": fmean(
            update["mean_total_loss"] for update in updates
        ),
        "first_update": updates[0],
        "last_update": updates[-1],
    }


def summarize_directory(results_dir: Path) -> dict[str, Any]:
    runs = {
        name: load_run(results_dir / f"{name}.json", expected)
        for name, expected in VARIANTS.items()
    }
    data_invariants = {
        (
            run["data"]["dataset"],
            run["data"]["seed"],
            run["data"]["window_count"],
            run["data"]["window_offsets_sha256"],
        )
        for run in runs.values()
    }
    if len(data_invariants) != 1:
        raise ValueError("D2 runs did not consume the same data windows")
    source_invariants = {
        tuple(sorted(run["source_sha256"].items())) for run in runs.values()
    }
    if len(source_invariants) != 1:
        raise ValueError("D2 runs did not use the same source revision")
    parameter_counts = {
        run["model_parameter_count"] for run in runs.values()
    }
    if len(parameter_counts) != 1:
        raise ValueError("D2 runs did not use the same model size")

    mechanism_keys = {
        "moe_capacity_factor",
        "moe_drop_tokens",
        "moe_balance_loss_weight",
        "moe_metrics_path",
    }
    common_configurations = {
        tuple(
            sorted(
                (key, json.dumps(value, sort_keys=True))
                for key, value in run["configuration"].items()
                if key not in mechanism_keys
            )
        )
        for run in runs.values()
    }
    if len(common_configurations) != 1:
        raise ValueError("D2 runs changed variables outside the mechanism sequence")

    unbounded_updates = runs["unbounded"]["updates"]
    observed_updates = runs["capacity_observe"]["updates"]
    observation_is_non_mutating = all(
        left["expert_counts"] == right["expert_counts"]
        and left["mean_lm_loss"] == right["mean_lm_loss"]
        and left["mean_total_loss"] == right["mean_total_loss"]
        for left, right in zip(unbounded_updates, observed_updates, strict=True)
    )
    if not observation_is_non_mutating:
        raise ValueError("capacity observation changed routing or training")

    summaries = {name: summarize_run(run) for name, run in runs.items()}
    dropped_last = summaries["capacity_drop"]["last_update"]
    balanced_last = summaries["balance_loss"]["last_update"]
    return {
        "schema_version": 1,
        "experiment": "D2 Top-1 MoE mechanism isolation summary",
        "invariants": {
            "same_data_windows": True,
            "same_source": True,
            "same_model_parameter_count": True,
            "model_parameter_count": next(iter(parameter_counts)),
            "only_declared_mechanisms_changed": True,
            "dataset": next(iter(data_invariants))[0],
            "data_seed": next(iter(data_invariants))[1],
            "window_count": next(iter(data_invariants))[2],
            "window_offsets_sha256": next(iter(data_invariants))[3],
            "capacity_observation_is_non_mutating": True,
        },
        "changed_variable_sequence": [
            {
                "variant": name,
                **expected,
            }
            for name, expected in VARIANTS.items()
        ],
        "variants": summaries,
        "last_update_balance_effect": {
            "max_to_mean_before": dropped_last["max_to_mean_load"],
            "max_to_mean_after": balanced_last["max_to_mean_load"],
            "load_cv_before": dropped_last[
                "load_coefficient_of_variation"
            ],
            "load_cv_after": balanced_last[
                "load_coefficient_of_variation"
            ],
            "drop_rate_before": dropped_last["drop_rate"],
            "drop_rate_after": balanced_last["drop_rate"],
            "lm_loss_before": dropped_last["mean_lm_loss"],
            "lm_loss_after": balanced_last["mean_lm_loss"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/d2_moe")
    )
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

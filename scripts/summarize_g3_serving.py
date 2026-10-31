#!/usr/bin/env python3
"""Validate and summarize independent G3 native serving runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from distributed_common import atomic_json_dump  # noqa: E402


GROUP_FIELDS = (
    "backend",
    "prefix_mode",
    "batch_size",
    "prompt_length",
    "output_length",
    "dtype",
    "decode_input_policy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("results/g_serving/g3_native/raw")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/g_serving/g3_native/summary.json"),
    )
    parser.add_argument("--expected-runs", type=int, default=3)
    return parser.parse_args()


def _group_key(run: dict[str, Any]) -> tuple[Any, ...]:
    configuration = run["configuration"]
    return tuple(configuration[field] for field in GROUP_FIELDS)


def _key_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(GROUP_FIELDS, key, strict=True))


def _load_runs(raw_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(raw_dir.rglob("run*.json"))
    if not paths:
        raise ValueError(f"no G3 run files found below {raw_dir}")
    return [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in paths
    ]


def summarize_runs(
    path_runs: list[tuple[Path, dict[str, Any]]], *, expected_runs: int
) -> dict[str, Any]:
    if expected_runs <= 0:
        raise ValueError("expected_runs must be positive")
    grouped: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = {}
    for path, run in path_runs:
        if run.get("stage") != "G3" or run.get("benchmark") != "native_serving":
            raise ValueError(f"not an G3 native serving result: {path}")
        if run.get("status") != "pass" or run.get("evidence_class") != "formal_gpu":
            raise ValueError(f"non-formal or failed result in formal summary: {path}")
        grouped.setdefault(_group_key(run), []).append((path, run))

    variants: list[dict[str, Any]] = []
    by_comparison: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        members = sorted(grouped[key], key=lambda item: item[1]["run_id"])
        if len(members) != expected_runs:
            raise ValueError(
                f"{_key_dict(key)} expected {expected_runs} runs, found {len(members)}"
            )
        run_ids = [run["run_id"] for _, run in members]
        if run_ids != [f"run{index}" for index in range(1, expected_runs + 1)]:
            raise ValueError(f"incomplete run IDs for {_key_dict(key)}: {run_ids}")
        runs = [run for _, run in members]
        invariants = {
            json.dumps(
                {
                    "artifact": run["artifact"],
                    "workload": run["workload"],
                    "environment": run["environment"],
                    "source_sha256": run["version"]["source_sha256"],
                    "configuration": run["configuration"],
                },
                sort_keys=True,
            )
            for run in runs
        }
        if len(invariants) != 1:
            raise ValueError(f"formal invariants differ for {_key_dict(key)}")
        if not all(run["measurement"]["outputs_deterministic"] for run in runs):
            raise ValueError(f"non-deterministic output for {_key_dict(key)}")
        if not all(run["measurement"]["predictions_deterministic"] for run in runs):
            raise ValueError(f"non-deterministic predictions for {_key_dict(key)}")
        output_hashes = {
            tuple(run["measurement"]["output_token_ids_sha256"]) for run in runs
        }
        if len(output_hashes) != 1:
            raise ValueError(f"output hashes differ across runs for {_key_dict(key)}")

        variant = {
            **_key_dict(key),
            "artifact_sha256": runs[0]["artifact"]["sha256"],
            "prompt_token_ids_sha256": runs[0]["workload"][
                "prompt_token_ids_sha256"
            ],
            "source_sha256": runs[0]["version"]["source_sha256"],
            "environment": runs[0]["environment"],
            "output_token_ids_sha256": list(next(iter(output_hashes))),
            "run_files": [path.as_posix() for path, _ in members],
            "run_ids": run_ids,
            "median_ttft_ms_by_run": [
                run["measurement"]["ttft_p50_ms"] for run in runs
            ],
            "median_of_run_median_ttft_ms": median(
                run["measurement"]["ttft_p50_ms"] for run in runs
            ),
            "median_tpot_ms_by_run": [
                run["measurement"]["tpot_p50_ms"] for run in runs
            ],
            "median_of_run_median_tpot_ms": median(
                run["measurement"]["tpot_p50_ms"] for run in runs
            ),
            "median_e2e_wall_ms_by_run": [
                run["measurement"]["e2e_wall_p50_ms"] for run in runs
            ],
            "median_of_run_median_e2e_wall_ms": median(
                run["measurement"]["e2e_wall_p50_ms"] for run in runs
            ),
            "output_tokens_per_second_by_run": [
                run["measurement"]["output_tokens_per_second"] for run in runs
            ],
            "median_output_tokens_per_second": median(
                run["measurement"]["output_tokens_per_second"] for run in runs
            ),
            "median_peak_allocated_bytes": median(
                run["measurement"]["peak_allocated_bytes"] for run in runs
            ),
            "median_peak_reserved_bytes": median(
                run["measurement"]["peak_reserved_bytes"] for run in runs
            ),
            "context_overflow": runs[0]["configuration"]["context_overflow"],
            "outputs_deterministic": True,
        }
        variants.append(variant)
        comparison_key = key[1:]
        by_comparison.setdefault(comparison_key, {})[key[0]] = variant

    comparisons: list[dict[str, Any]] = []
    for key in sorted(by_comparison, key=lambda value: tuple(str(item) for item in value)):
        sides = by_comparison[key]
        if set(sides) != {"naive", "kv_cache"}:
            raise ValueError(f"missing naive/KV pair for {key}")
        naive = sides["naive"]
        cached = sides["kv_cache"]
        cross_backend_invariants = (
            "artifact_sha256",
            "prompt_token_ids_sha256",
            "source_sha256",
            "environment",
            "output_token_ids_sha256",
        )
        if any(naive[field] != cached[field] for field in cross_backend_invariants):
            raise ValueError(f"naive/KV invariants differ for {key}")
        comparisons.append(
            {
                **dict(zip(GROUP_FIELDS[1:], key, strict=True)),
                "context_overflow": naive["context_overflow"],
                "kv_to_naive_output_throughput_ratio": (
                    cached["median_output_tokens_per_second"]
                    / naive["median_output_tokens_per_second"]
                ),
                "naive_to_kv_ttft_ratio": (
                    naive["median_of_run_median_ttft_ms"]
                    / cached["median_of_run_median_ttft_ms"]
                ),
                "naive_to_kv_tpot_ratio": (
                    naive["median_of_run_median_tpot_ms"]
                    / cached["median_of_run_median_tpot_ms"]
                ),
                "naive_to_kv_e2e_ratio": (
                    naive["median_of_run_median_e2e_wall_ms"]
                    / cached["median_of_run_median_e2e_wall_ms"]
                ),
                "kv_minus_naive_peak_allocated_bytes": (
                    cached["median_peak_allocated_bytes"]
                    - naive["median_peak_allocated_bytes"]
                ),
            }
        )

    return {
        "schema_version": 1,
        "stage": "G3",
        "benchmark": "native_serving_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "independent_runs_per_variant": expected_runs,
        "variant_count": len(variants),
        "comparison_count": len(comparisons),
        "invariants": {
            "formal_gpu_only": True,
            "same_artifact_workload_environment_source_per_variant": True,
            "deterministic_outputs_across_repeats_and_runs": True,
            "median_of_independent_run_medians": True,
        },
        "variants": variants,
        "comparisons": comparisons,
    }


def summarize_directory(
    raw_dir: Path, output: Path, *, expected_runs: int
) -> dict[str, Any]:
    summary = summarize_runs(_load_runs(raw_dir), expected_runs=expected_runs)
    atomic_json_dump(summary, output)
    return summary


def main() -> None:
    args = parse_args()
    summary = summarize_directory(
        args.raw_dir, args.output, expected_runs=args.expected_runs
    )
    print(
        f"G3 summary: {summary['variant_count']} variants, "
        f"{summary['comparison_count']} naive/KV comparisons -> {args.output}"
    )


if __name__ == "__main__":
    main()


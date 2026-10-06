#!/usr/bin/env python3
"""Summarize successful and failed four-GPU C-stage capacity probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANT_ORDER = ("native_ddp", "deepspeed_zero2", "deepspeed_zero3", "fsdp2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/c_capacity")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def variant_for(path: Path, result: dict) -> str | None:
    if result.get("variant"):
        return result["variant"]
    if result.get("backend") == "native_ddp" or "ddp" in path.parts:
        return "native_ddp"
    return None


def parameter_count(result: dict) -> int | None:
    configuration = result.get("configuration", {})
    if configuration.get("model_parameter_count") is not None:
        return int(configuration["model_parameter_count"])
    model = result.get("model", {})
    if model.get("parameter_count") is not None:
        return int(model["parameter_count"])
    return None


def load_points(root: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    successes: dict[str, list[dict]] = {}
    failures: dict[str, list[dict]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        variant = variant_for(path, result)
        count = parameter_count(result)
        if variant is None or count is None:
            continue
        status = result.get("status")
        world_size = result.get("configuration", {}).get("world_size")
        if status == "benchmark_complete" and world_size == 4:
            measurement = result["measurement"]
            ranks = measurement["per_rank"]
            if "peak_allocated_bytes" in ranks[0]:
                allocated = max(item["peak_allocated_bytes"] for item in ranks)
                reserved = max(item["peak_reserved_bytes"] for item in ranks)
            else:
                allocated = int(
                    max(item["peak_allocated_mib"] for item in ranks) * 1024**2
                )
                reserved = int(
                    max(item["peak_reserved_mib"] for item in ranks) * 1024**2
                )
            cpu_values = [
                item.get("peak_process_rss_bytes")
                for item in ranks
                if item.get("peak_process_rss_bytes") is not None
            ]
            successes.setdefault(variant, []).append(
                {
                    "parameter_count": count,
                    "run_id": result.get("run_id"),
                    "path": str(path),
                    "configuration": result.get("configuration", {}),
                    "median_step_ms": measurement["median_slowest_rank_step_ms"],
                    "global_tokens_per_second": measurement[
                        "global_tokens_per_second_from_median"
                    ],
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "max_rank_peak_process_rss_bytes": (
                        max(cpu_values) if cpu_values else None
                    ),
                    "all_losses_finite": measurement["all_losses_finite"],
                    "all_window_hashes_match_plan": measurement[
                        "all_window_hashes_match_plan"
                    ],
                }
            )
        elif status in {"capacity_failure", "oom", "operational_failure"}:
            failure = result.get("failure", {})
            failures.setdefault(variant, []).append(
                {
                    "parameter_count": count,
                    "run_id": result.get("run_id"),
                    "path": str(path),
                    "configuration": result.get("configuration", {}),
                    "failure_type": failure.get("type", status),
                    "failure_phase": failure.get("phase"),
                    "elapsed_seconds": failure.get("elapsed_seconds"),
                    "resource_snapshot": failure.get("resource_snapshot"),
                }
            )
    return successes, failures


def summarize_variant(
    variant: str, successes: list[dict], failures: list[dict]
) -> dict:
    successes = sorted(successes, key=lambda item: item["parameter_count"])
    failures = sorted(failures, key=lambda item: item["parameter_count"])
    maximum = successes[-1] if successes else None
    first_failure = None
    if maximum is not None:
        first_failure = next(
            (
                item
                for item in failures
                if item["parameter_count"] > maximum["parameter_count"]
            ),
            None,
        )
    elif failures:
        first_failure = failures[0]
    return {
        "variant": variant,
        "maximum_tested_success": maximum,
        "first_tested_failure_above_success": first_failure,
        "success_is_lower_bound_without_failure": (
            maximum is not None and first_failure is None
        ),
        "successful_points": successes,
        "failure_points": failures,
    }


def main() -> None:
    args = parse_args()
    successes, failures = load_points(args.results_dir)
    variants = [
        variant
        for variant in VARIANT_ORDER
        if variant in successes or variant in failures
    ]
    summaries = [
        summarize_variant(
            variant,
            successes.get(variant, []),
            failures.get(variant, []),
        )
        for variant in variants
    ]
    result = {
        "schema_version": 1,
        "analysis": "C6/C7 four-GPU capacity summary",
        "variants": summaries,
        "notes": [
            "A success without a higher failure is a tested lower bound, not an exact maximum.",
            "Capacity probes use short 2-warmup/5-measurement runs and are not formal performance rankings.",
        ],
    }
    output = args.output or args.results_dir / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("C capacity summary")
    for summary in summaries:
        success = summary["maximum_tested_success"]
        failure = summary["first_tested_failure_above_success"]
        print(
            f"  {summary['variant']}: "
            f"success={success['parameter_count'] if success else None}, "
            f"failure={failure['parameter_count'] if failure else None}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

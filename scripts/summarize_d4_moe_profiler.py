"""Summarize D4 eager/compiled MoE profiler diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROUTING_OPERATIONS = (
    "aten::nonzero",
    "aten::bincount",
    "aten::index_select",
    "aten::index_copy",
)


def parse_time_total(table_text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s+([0-9.]+)(ms|s|us)", table_text)
    if match is None:
        raise ValueError(f"missing {label} in profiler table")
    value = float(match.group(1))
    return value * {"us": 0.001, "ms": 1.0, "s": 1000.0}[match.group(2)]


def aggregate_named(rows: list[dict[str, Any]], name: str) -> dict[str, float | int]:
    selected = [row for row in rows if row["name"] == name]
    return {
        "count": sum(row["count"] for row in selected),
        "self_cpu_time_ms": sum(row["self_cpu_time_us"] for row in selected) / 1000,
        "self_device_time_ms": (
            sum(row["self_device_time_us"] for row in selected) / 1000
        ),
        "device_time_total_ms": (
            sum(row["device_time_total_us"] for row in selected) / 1000
        ),
    }


def read_profile(root: Path, name: str) -> dict[str, Any]:
    rows = json.loads(
        (root / f"{name}_key_averages.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (root / f"{name}_metadata.json").read_text(encoding="utf-8")
    )
    table = (root / f"{name}_key_averages.txt").read_text(encoding="utf-8")
    graph_names = {
        row["name"] for row in rows if "Call CompiledFxGraph" in row["name"]
    }
    triton_names = {row["name"] for row in rows if row["name"].startswith("triton_")}
    cuda_launch = aggregate_named(rows, "cudaLaunchKernel")
    cu_launch = aggregate_named(rows, "cuLaunchKernel")
    return {
        "run_id": name,
        "compile": metadata["configuration"]["compile"],
        "active_steps": metadata["schedule"]["active_steps"],
        "all_losses_finite": metadata["correctness"]["all_losses_finite"],
        "self_cpu_time_total_ms": parse_time_total(table, "Self CPU time total"),
        "self_cuda_time_total_ms": parse_time_total(table, "Self CUDA time total"),
        "peak_allocated_mib": metadata["memory"]["peak_allocated_mib"],
        "peak_reserved_mib": metadata["memory"]["peak_reserved_mib"],
        "combined_launch_api_count": cuda_launch["count"] + cu_launch["count"],
        "aten_mm": aggregate_named(rows, "aten::mm"),
        "aten_copy": aggregate_named(rows, "aten::copy_"),
        "routing_operations": {
            operation: aggregate_named(rows, operation)
            for operation in ROUTING_OPERATIONS
        },
        "compiled_fx_graph_event_names": len(graph_names),
        "triton_kernel_event_names": len(triton_names),
        "configuration": {
            field: metadata["configuration"][field]
            for field in (
                "dataset",
                "batch_size",
                "block_size",
                "gradient_accumulation_steps",
                "n_layer",
                "n_head",
                "n_embd",
                "dtype",
                "moe_num_experts",
                "moe_layer_index",
                "moe_capacity_factor",
                "moe_drop_tokens",
                "moe_balance_loss_weight",
                "profiler_data_seed",
            )
        },
    }


def percent_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0 if after == 0 else float("inf")
    return (after / before - 1.0) * 100


def summarize(root: Path) -> dict[str, Any]:
    eager = read_profile(root, "moe_eager")
    compiled = read_profile(root, "moe_compiled")
    if eager["compile"] or not compiled["compile"]:
        raise ValueError("expected eager compile=False and compiled compile=True")
    if eager["configuration"] != compiled["configuration"]:
        raise ValueError("profiler workload differs beyond compile mode")
    if eager["active_steps"] != compiled["active_steps"]:
        raise ValueError("profiler active-step counts differ")
    if not eager["all_losses_finite"] or not compiled["all_losses_finite"]:
        raise ValueError("a profiler run contains a non-finite loss")

    routing_count_changes = {
        operation: percent_change(
            eager["routing_operations"][operation]["count"],
            compiled["routing_operations"][operation]["count"],
        )
        for operation in ROUTING_OPERATIONS
    }
    return {
        "schema_version": 1,
        "experiment": "D4 Top-1 MoE eager/compiled profiler diagnosis",
        "warning": (
            "Profiler totals contain instrumentation overhead and overlapping "
            "scope rows; formal performance comes from the no-Profiler benchmark."
        ),
        "invariants": {
            "same_workload_except_compile": True,
            "active_steps": eager["active_steps"],
            "all_losses_finite": True,
        },
        "eager": eager,
        "compiled": compiled,
        "comparison": {
            "self_cpu_time_change_percent": percent_change(
                eager["self_cpu_time_total_ms"], compiled["self_cpu_time_total_ms"]
            ),
            "self_cuda_time_change_percent": percent_change(
                eager["self_cuda_time_total_ms"],
                compiled["self_cuda_time_total_ms"],
            ),
            "combined_launch_api_count_change_percent": percent_change(
                eager["combined_launch_api_count"],
                compiled["combined_launch_api_count"],
            ),
            "aten_copy_call_change_percent": percent_change(
                eager["aten_copy"]["count"], compiled["aten_copy"]["count"]
            ),
            "aten_mm_self_device_time_change_percent": percent_change(
                eager["aten_mm"]["self_device_time_ms"],
                compiled["aten_mm"]["self_device_time_ms"],
            ),
            "routing_operation_count_change_percent": routing_count_changes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/d4_moe_compile/profiler"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(args.results_dir)
    output = args.output or args.results_dir / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = result["comparison"]
    print(
        "D4 MoE profiler: "
        f"CUDA self time {comparison['self_cuda_time_change_percent']:+.1f}%, "
        f"CPU self time {comparison['self_cpu_time_change_percent']:+.1f}%, "
        "launch API calls "
        f"{comparison['combined_launch_api_count_change_percent']:+.1f}%"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

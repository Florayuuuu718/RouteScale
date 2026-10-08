"""Compare compiled ragged-loop and padded-Triton MoE profiler runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_d4_moe_profiler import (
        aggregate_named,
        parse_time_total,
        percent_change,
    )
except ModuleNotFoundError:  # direct execution adds scripts/, not the repo root
    from summarize_d4_moe_profiler import (
        aggregate_named,
        parse_time_total,
        percent_change,
    )


OPERATIONS = (
    "aten::mm",
    "aten::bmm",
    "aten::copy_",
    "aten::nonzero",
    "aten::bincount",
    "aten::index_select",
    "aten::index_copy",
    "aten::index_add",
)

WORKLOAD_FIELDS = (
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
    "compile",
    "profiler_data_seed",
)


def count_percent_change(before: int, after: int) -> float | None:
    """Return a finite percentage, or None when an operation is newly added."""
    if before == 0:
        return 0.0 if after == 0 else None
    return percent_change(before, after)


def read_profile(root: Path, run_id: str) -> dict[str, Any]:
    rows = json.loads(
        (root / f"{run_id}_key_averages.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (root / f"{run_id}_metadata.json").read_text(encoding="utf-8")
    )
    table = (root / f"{run_id}_key_averages.txt").read_text(encoding="utf-8")
    launch_count = sum(
        aggregate_named(rows, name)["count"]
        for name in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx")
    )
    grouped_rows = [
        row for row in rows if row["name"].startswith("_fixed_grouped_gemm_kernel")
    ]
    return {
        "run_id": run_id,
        "dispatch_backend": metadata["configuration"][
            "moe_dispatch_backend"
        ],
        "active_steps": metadata["schedule"]["active_steps"],
        "all_losses_finite": metadata["correctness"]["all_losses_finite"],
        "self_cpu_time_total_ms": parse_time_total(
            table, "Self CPU time total"
        ),
        "self_cuda_time_total_ms": parse_time_total(
            table, "Self CUDA time total"
        ),
        "peak_allocated_mib": metadata["memory"]["peak_allocated_mib"],
        "peak_reserved_mib": metadata["memory"]["peak_reserved_mib"],
        "combined_launch_api_count": launch_count,
        "grouped_gemm_event_count": sum(
            row["count"] for row in grouped_rows
        ),
        "grouped_gemm_event_names": sorted(
            {row["name"] for row in grouped_rows}
        ),
        "operations": {
            operation: aggregate_named(rows, operation)
            for operation in OPERATIONS
        },
        "workload": {
            field: metadata["configuration"][field]
            for field in WORKLOAD_FIELDS
        },
    }


def summarize(root: Path) -> dict[str, Any]:
    loop = read_profile(root, "loop_compiled")
    padded = read_profile(root, "padded_triton_compiled")
    if loop["dispatch_backend"] != "loop":
        raise ValueError("expected loop baseline")
    if padded["dispatch_backend"] != "padded_triton":
        raise ValueError("expected padded_triton candidate")
    if loop["workload"] != padded["workload"]:
        raise ValueError("profiler workload differs beyond dispatch backend")
    if loop["active_steps"] != padded["active_steps"]:
        raise ValueError("active profiler step counts differ")
    if not loop["all_losses_finite"] or not padded["all_losses_finite"]:
        raise ValueError("profiler run contains non-finite loss")
    return {
        "schema_version": 1,
        "experiment": "E-stage compiled loop versus padded Triton profiler",
        "warning": (
            "Profiler totals include instrumentation overhead; formal "
            "performance comes from the no-Profiler benchmark."
        ),
        "invariants": {
            "same_workload_except_dispatch_backend": True,
            "active_steps": loop["active_steps"],
            "all_losses_finite": True,
        },
        "loop_compiled": loop,
        "padded_triton_compiled": padded,
        "comparison": {
            "self_cpu_time_change_percent": percent_change(
                loop["self_cpu_time_total_ms"],
                padded["self_cpu_time_total_ms"],
            ),
            "self_cuda_time_change_percent": percent_change(
                loop["self_cuda_time_total_ms"],
                padded["self_cuda_time_total_ms"],
            ),
            "launch_api_count_change_percent": percent_change(
                loop["combined_launch_api_count"],
                padded["combined_launch_api_count"],
            ),
            "operation_count_change_percent": {
                operation: count_percent_change(
                    loop["operations"][operation]["count"],
                    padded["operations"][operation]["count"],
                )
                for operation in OPERATIONS
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/e_triton_moe/profiler"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(args.results_dir)
    output = args.output or args.results_dir / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["comparison"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

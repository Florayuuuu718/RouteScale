#!/usr/bin/env python3
"""Quantify NCCL-kernel overlap from existing C-stage Chrome traces."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import median
from typing import Iterable


Interval = tuple[float, float]


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_duration(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_duration(
    left: Iterable[Interval], right: Iterable[Interval]
) -> float:
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    total = 0.0
    i = 0
    j = 0
    while i < len(left_merged) and j < len(right_merged):
        left_start, left_end = left_merged[i]
        right_start, right_end = right_merged[j]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            i += 1
        else:
            j += 1
    return total


def load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    events = value.get("traceEvents", []) if isinstance(value, dict) else value
    if not isinstance(events, list):
        raise TypeError(f"{path}: Chrome trace events must be a list")
    return events


def complete_interval(event: dict) -> Interval | None:
    if event.get("ph") != "X":
        return None
    try:
        start = float(event["ts"])
        duration = float(event["dur"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, start + duration) if duration > 0 else None


def is_gpu_kernel(event: dict) -> bool:
    category = str(event.get("cat", "")).casefold()
    return "kernel" in category


def analyze_trace(path: Path) -> dict:
    nccl_intervals: list[Interval] = []
    compute_intervals: list[Interval] = []
    backward_intervals: list[Interval] = []
    nccl_names: dict[str, int] = {}
    for event in load_trace(path):
        interval = complete_interval(event)
        if interval is None:
            continue
        name = str(event.get("name", ""))
        lowered = name.casefold()
        if is_gpu_kernel(event):
            if "nccl" in lowered:
                nccl_intervals.append(interval)
                nccl_names[name] = nccl_names.get(name, 0) + 1
            else:
                compute_intervals.append(interval)
        if lowered == "backward":
            backward_intervals.append(interval)

    nccl_time = interval_duration(nccl_intervals)
    compute_time = interval_duration(compute_intervals)
    direct_overlap = intersection_duration(nccl_intervals, compute_intervals)
    backward_scope_overlap = intersection_duration(
        nccl_intervals, backward_intervals
    )
    return {
        "trace": str(path),
        "nccl_kernel_count": len(nccl_intervals),
        "compute_kernel_count": len(compute_intervals),
        "backward_scope_count": len(backward_intervals),
        "nccl_kernel_union_time_us": nccl_time,
        "compute_kernel_union_time_us": compute_time,
        "nccl_compute_overlap_us": direct_overlap,
        "nccl_compute_overlap_ratio": (
            direct_overlap / nccl_time if nccl_time else None
        ),
        "nccl_within_backward_scope_us": backward_scope_overlap,
        "nccl_within_backward_scope_ratio": (
            backward_scope_overlap / nccl_time if nccl_time else None
        ),
        "nccl_kernel_names": [
            {"name": name, "count": count}
            for name, count in sorted(nccl_names.items())
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = []
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = manifest.get("variant") or "native_ddp"
        ranks = []
        for rank_metadata in manifest["ranks"]:
            trace_path = Path(rank_metadata["artifacts"]["trace"])
            if not trace_path.is_absolute():
                trace_path = args.repo_root / trace_path
            analysis = analyze_trace(trace_path)
            analysis["rank"] = int(rank_metadata["rank"])
            ranks.append(analysis)
        ratios = [
            rank["nccl_compute_overlap_ratio"]
            for rank in ranks
            if rank["nccl_compute_overlap_ratio"] is not None
        ]
        backward_ratios = [
            rank["nccl_within_backward_scope_ratio"]
            for rank in ranks
            if rank["nccl_within_backward_scope_ratio"] is not None
        ]
        variants.append(
            {
                "variant": variant,
                "world_size": manifest["world_size"],
                "all_losses_finite": manifest["all_losses_finite"],
                "all_window_hashes_match_plan": manifest[
                    "all_window_hashes_match_plan"
                ],
                "median_nccl_compute_overlap_ratio": (
                    median(ratios) if ratios else None
                ),
                "median_nccl_within_backward_scope_ratio": (
                    median(backward_ratios) if backward_ratios else None
                ),
                "ranks": sorted(ranks, key=lambda item: item["rank"]),
            }
        )

    result = {
        "schema_version": 1,
        "analysis": "C-stage NCCL/compute overlap from Chrome traces",
        "variants": variants,
        "method": {
            "direct_overlap": (
                "intersection of GPU NCCL kernel intervals and non-NCCL GPU "
                "kernel intervals on the same rank"
            ),
            "backward_scope": (
                "NCCL kernel time occurring inside host record_function "
                "regions named backward"
            ),
            "timing_unit": "microseconds",
        },
        "limitations": [
            "Profiler instrumentation changes timings, so ratios explain scheduling rather than formal throughput.",
            "Backward host scope is a proxy; direct GPU-kernel overlap is the stronger concurrency signal.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("C trace overlap summary")
    for variant in variants:
        direct = variant["median_nccl_compute_overlap_ratio"]
        backward = variant["median_nccl_within_backward_scope_ratio"]
        print(
            f"  {variant['variant']}: "
            f"direct={direct:.1%} " if direct is not None else
            f"  {variant['variant']}: direct=n/a ",
            end="",
        )
        print(f"backward-scope={backward:.1%}" if backward is not None else "backward-scope=n/a")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

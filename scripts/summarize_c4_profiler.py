#!/usr/bin/env python3
"""Summarize collective events from per-rank C4 PyTorch Profiler output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COLLECTIVE_PATTERNS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "reducescatter",
    "all_gather",
    "allgather",
    "c10d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/c_ddp/profiler")
    )
    parser.add_argument("--run-id", default="4gpu")
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def is_collective(name: str) -> bool:
    lowered = name.casefold()
    return any(pattern in lowered for pattern in COLLECTIVE_PATTERNS)


def main() -> None:
    args = parse_args()
    manifest_path = args.results_dir / f"{args.run_id}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["world_size"] != args.expected_ranks:
        raise RuntimeError(
            f"expected {args.expected_ranks} ranks, found {manifest['world_size']}"
        )
    if not manifest["all_losses_finite"]:
        raise RuntimeError("profiler run contains non-finite losses")
    if not manifest["all_window_hashes_match_plan"]:
        raise RuntimeError("profiler window hashes do not match the plan")

    ranks = []
    for rank in range(args.expected_ranks):
        events_path = args.results_dir / (
            f"{args.run_id}_rank{rank}_key_averages.json"
        )
        events = json.loads(events_path.read_text(encoding="utf-8"))
        collectives = [event for event in events if is_collective(event["name"])]
        collectives.sort(
            key=lambda event: event["self_device_time_us"], reverse=True
        )
        metadata_path = args.results_dir / f"{args.run_id}_rank{rank}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        active_steps = metadata["schedule"]["active_steps"]
        self_device_total = sum(
            event["self_device_time_us"] for event in collectives
        )
        ranks.append(
            {
                "rank": rank,
                "active_steps": active_steps,
                "collective_event_count": sum(
                    event["count"] for event in collectives
                ),
                "collective_self_device_time_us": self_device_total,
                "collective_self_device_time_us_per_active_update": (
                    self_device_total / active_steps
                ),
                "top_collective_events": collectives[:20],
                "trace": metadata["artifacts"]["trace"],
            }
        )

    result = {
        "schema_version": 1,
        "analysis": "C4 per-rank collective event summary",
        "status": "pass",
        "run_id": args.run_id,
        "world_size": args.expected_ranks,
        "all_losses_finite": True,
        "all_window_hashes_match_plan": True,
        "ranks": ranks,
        "overlap_analysis": {
            "automatically_decided": False,
            "reason": (
                "aggregate key averages do not preserve event concurrency; "
                "inspect backward and collective lanes in each Chrome trace"
            ),
        },
        "warning": (
            "Profiler timings include instrumentation overhead and are not "
            "formal throughput results."
        ),
    }
    output = args.output or args.results_dir / f"{args.run_id}_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("C4 collective summary")
    for rank in ranks:
        print(
            f"  rank {rank['rank']}: "
            f"{rank['collective_event_count']} collective events, "
            f"{rank['collective_self_device_time_us_per_active_update'] / 1000:.3f} "
            "ms self device time / active update"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

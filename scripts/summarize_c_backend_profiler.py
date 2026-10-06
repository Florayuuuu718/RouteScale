#!/usr/bin/env python3
"""Summarize collective families from C6/C7 per-rank profiler artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collective_family(name: str) -> str | None:
    lowered = name.casefold().replace("-", "_")
    compact = lowered.replace("_", "")
    if "reducescatter" in compact:
        return "reduce_scatter"
    if "allgather" in compact:
        return "all_gather"
    if "allreduce" in compact:
        return "all_reduce"
    if "nccl" in lowered or "c10d" in lowered:
        return "other_collective"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/c_backend_profiler"),
    )
    parser.add_argument("--run-id", default="4gpu")
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument(
        "--variants",
        nargs="*",
        help="variant directories; by default discover all manifests",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize_variant(
    root: Path, variant: str, run_id: str, expected_ranks: int
) -> dict:
    directory = root / variant
    manifest_path = directory / f"{run_id}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["variant"] != variant:
        raise RuntimeError(f"{manifest_path} contains {manifest['variant']}")
    if manifest["world_size"] != expected_ranks:
        raise RuntimeError(
            f"{variant}: expected {expected_ranks} ranks, "
            f"found {manifest['world_size']}"
        )
    if not manifest["all_losses_finite"]:
        raise RuntimeError(f"{variant}: non-finite profiler loss")
    if not manifest["all_window_hashes_match_plan"]:
        raise RuntimeError(f"{variant}: profiler window hash mismatch")

    ranks = []
    observed_families: set[str] = set()
    for rank in range(expected_ranks):
        events_path = directory / f"{run_id}_rank{rank}_key_averages.json"
        metadata_path = directory / f"{run_id}_rank{rank}_metadata.json"
        events = json.loads(events_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict]] = {}
        for event in events:
            family = collective_family(event["name"])
            if family is not None:
                grouped.setdefault(family, []).append(event)
                observed_families.add(family)
        families = {}
        for family, rows in sorted(grouped.items()):
            rows.sort(
                key=lambda item: item["self_device_time_us"], reverse=True
            )
            families[family] = {
                "event_count": sum(item["count"] for item in rows),
                "self_device_time_us": sum(
                    item["self_device_time_us"] for item in rows
                ),
                "events": rows,
            }
        ranks.append(
            {
                "rank": rank,
                "active_steps": metadata["schedule"]["active_steps"],
                "families": families,
                "trace": metadata["artifacts"]["trace"],
            }
        )
    return {
        "variant": variant,
        "backend": manifest["backend"],
        "world_size": expected_ranks,
        "observed_collective_families": sorted(observed_families),
        "ranks": ranks,
    }


def main() -> None:
    args = parse_args()
    if args.expected_ranks <= 0:
        raise ValueError("expected-ranks must be positive")
    variants = args.variants
    if not variants:
        variants = sorted(
            path.name
            for path in args.results_dir.iterdir()
            if (path / f"{args.run_id}_manifest.json").is_file()
        )
    if not variants:
        raise RuntimeError(f"no profiler manifests found in {args.results_dir}")
    summaries = [
        summarize_variant(
            args.results_dir,
            variant,
            args.run_id,
            args.expected_ranks,
        )
        for variant in variants
    ]
    result = {
        "schema_version": 1,
        "analysis": "C6/C7 backend collective-family profiler summary",
        "run_id": args.run_id,
        "world_size": args.expected_ranks,
        "variants": summaries,
        "all_losses_finite": True,
        "all_window_hashes_match_plan": True,
        "overlap_analysis": {
            "automatically_decided": False,
            "reason": (
                "key averages do not preserve concurrency; inspect backward "
                "and collective lanes in the Chrome traces"
            ),
        },
        "warning": (
            "Profiler timings include instrumentation overhead. Wrapper and "
            "kernel events are preserved separately and must not be summed "
            "as independent communication work."
        ),
    }
    output = args.output or args.results_dir / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("C6/C7 backend profiler summary")
    for summary in summaries:
        print(
            f"  {summary['variant']}: "
            + ", ".join(summary["observed_collective_families"])
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

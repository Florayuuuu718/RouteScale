#!/usr/bin/env python3
"""Summarize quantitative DeepSpeed/FSDP2 checkpoint evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/c_checkpoint_metrics"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    verifications = []
    for path in sorted(args.results_dir.rglob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result.get("metrics", {})
        checkpoint = result.get("artifacts", {}).get("checkpoint")
        if checkpoint and "checkpoint_save_seconds_slowest_rank" in metrics:
            runs.append(
                {
                    "path": str(path),
                    "backend": result["backend"],
                    "zero_stage": result.get("zero_stage"),
                    "checkpoint_tag": result.get("artifacts", {}).get(
                        "checkpoint_tag"
                    ),
                    "file_count": checkpoint["file_count"],
                    "total_bytes": checkpoint["total_bytes"],
                    "files": checkpoint["files"],
                    "save_seconds_slowest_rank": metrics[
                        "checkpoint_save_seconds_slowest_rank"
                    ],
                    "load_seconds_slowest_rank": metrics.get(
                        "checkpoint_load_seconds_slowest_rank"
                    ),
                    "max_rank_peak_process_rss_bytes": max(
                        rank["peak_process_rss_bytes"]
                        for rank in metrics["per_rank"]
                    ),
                }
            )
        if result.get("status") == "pass" and "merge_seconds" in result:
            verifications.append(
                {
                    "path": str(path),
                    "zero_stage": result["zero_stage"],
                    "checkpoint_file_count": result["checkpoint"]["file_count"],
                    "checkpoint_total_bytes": result["checkpoint"]["total_bytes"],
                    "merge_seconds": result["merge_seconds"],
                    "strict_load_seconds": result["strict_load_seconds"],
                    "peak_process_rss_bytes": result["peak_process_rss_bytes"],
                    "merged_output_bytes": result["merged_output_bytes"],
                    "model_state_sha256": result["model_state_sha256"],
                }
            )
    if not runs:
        raise RuntimeError(f"no checkpoint run results in {args.results_dir}")
    result = {
        "schema_version": 1,
        "analysis": "C6/C7 checkpoint quantitative summary",
        "runs": runs,
        "fp32_verifications": verifications,
    }
    output = args.output or args.results_dir / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("C checkpoint summary")
    for run in runs:
        stage = (
            f" ZeRO-{run['zero_stage']}" if run["zero_stage"] is not None else ""
        )
        print(
            f"  {run['backend']}{stage}: {run['file_count']} files, "
            f"{run['total_bytes'] / 1024**2:.1f} MiB, "
            f"save {run['save_seconds_slowest_rank']:.3f}s, "
            f"load {run['load_seconds_slowest_rank']}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

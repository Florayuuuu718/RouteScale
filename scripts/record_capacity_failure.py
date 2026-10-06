#!/usr/bin/env python3
"""Create a structured capacity-failure record from an experiment log."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parameter-count", required=True, type=int)
    parser.add_argument("--n-layer", required=True, type=int)
    parser.add_argument("--n-head", required=True, type=int)
    parser.add_argument("--n-embd", required=True, type=int)
    parser.add_argument(
        "--failure-type",
        required=True,
        choices=("cuda_oom", "operational_hang", "timeout", "other"),
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--gpu-memory-used-mib", type=float)
    parser.add_argument("--gpu-memory-total-mib", type=float)
    parser.add_argument("--torch-allocated-mib", type=float)
    parser.add_argument("--torch-reserved-unallocated-mib", type=float)
    parser.add_argument("--requested-allocation-mib", type=float)
    parser.add_argument("--max-rank-process-rss-mib", type=float)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_bytes = args.log.read_bytes()
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "capacity_failure",
        "variant": args.variant,
        "run_id": args.run_id,
        "configuration": {
            "world_size": 4,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "model_parameter_count": args.parameter_count,
            "batch_size_per_rank": 8,
            "block_size": 512,
            "dtype": "bfloat16",
            "offload": False,
        },
        "failure": {
            "type": args.failure_type,
            "phase": args.phase,
            "elapsed_seconds": args.elapsed_seconds,
            "resource_snapshot": {
                "gpu_index": args.gpu_index,
                "gpu_memory_used_bytes": (
                    int(args.gpu_memory_used_mib * 1024**2)
                    if args.gpu_memory_used_mib is not None
                    else None
                ),
                "gpu_memory_total_bytes": (
                    int(args.gpu_memory_total_mib * 1024**2)
                    if args.gpu_memory_total_mib is not None
                    else None
                ),
                "torch_allocated_bytes": (
                    int(args.torch_allocated_mib * 1024**2)
                    if args.torch_allocated_mib is not None
                    else None
                ),
                "torch_reserved_unallocated_bytes": (
                    int(args.torch_reserved_unallocated_mib * 1024**2)
                    if args.torch_reserved_unallocated_mib is not None
                    else None
                ),
                "requested_allocation_bytes": (
                    int(args.requested_allocation_mib * 1024**2)
                    if args.requested_allocation_mib is not None
                    else None
                ),
                "max_rank_process_rss_bytes": (
                    int(args.max_rank_process_rss_mib * 1024**2)
                    if args.max_rank_process_rss_mib is not None
                    else None
                ),
            },
        },
        "artifacts": {
            "log": str(args.log),
            "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

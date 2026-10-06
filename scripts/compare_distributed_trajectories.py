#!/usr/bin/env python3
"""Compare native, DeepSpeed and FSDP2 deterministic smoke trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from distributed_common import atomic_json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--deepspeed-model-state", required=True)
    parser.add_argument("--fsdp2-metadata", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--parameter-atol", type=float, default=1e-7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native = torch.load(args.native_checkpoint, map_location="cpu", weights_only=False)
    deepspeed = torch.load(
        args.deepspeed_model_state, map_location="cpu", weights_only=False
    )
    with Path(args.fsdp2_metadata).open(encoding="utf-8") as handle:
        fsdp2 = json.load(handle)
    histories = {
        "native_ddp": [float(value) for value in native["loss_history"]],
        "deepspeed_zero0": [float(value) for value in deepspeed["loss_history"]],
        "fsdp2": [float(value) for value in fsdp2["loss_history"]],
    }
    lengths = {len(values) for values in histories.values()}
    if len(lengths) != 1:
        raise AssertionError(f"trajectory length mismatch: {lengths}")
    reference = histories["native_ddp"]
    max_differences = {
        backend: max(
            (abs(left - right) for left, right in zip(reference, values)),
            default=0.0,
        )
        for backend, values in histories.items()
    }
    hashes = {
        "native_ddp": native["global_window_prefix_sha256"],
        "deepspeed_zero0": deepspeed["global_window_prefix_sha256"],
        "fsdp2": fsdp2["global_window_prefix_sha256"],
    }
    native_model = native["model"]
    deepspeed_model = deepspeed["module"]
    if set(native_model) != set(deepspeed_model):
        raise AssertionError("native and DeepSpeed model state keys differ")
    parameter_differences = {
        name: float(
            (native_model[name].float() - deepspeed_model[name].float()).abs().max()
        )
        for name in native_model
    }
    worst_parameter = max(parameter_differences, key=parameter_differences.get)
    max_parameter_difference = parameter_differences[worst_parameter]
    passed = (
        all(value <= args.atol for value in max_differences.values())
        and len(set(hashes.values())) == 1
        and max_parameter_difference <= args.parameter_atol
    )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "updates_compared": len(reference),
        "absolute_tolerance": args.atol,
        "parameter_absolute_tolerance": args.parameter_atol,
        "loss_histories": histories,
        "max_absolute_loss_difference_vs_native": max_differences,
        "deepspeed_zero0_parameter_update_vs_native": {
            "max_absolute_difference": max_parameter_difference,
            "worst_parameter": worst_parameter,
            "within_tolerance": max_parameter_difference <= args.parameter_atol,
        },
        "global_window_prefix_sha256": hashes,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        atomic_json_dump(result, args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

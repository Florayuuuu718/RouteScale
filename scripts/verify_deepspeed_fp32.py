#!/usr/bin/env python3
"""Consolidate a ZeRO checkpoint and strictly load it into an ordinary GPT."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from distributed_common import (  # noqa: E402
    atomic_json_dump,
    atomic_torch_save,
    build_model,
    directory_artifact_stats,
    state_dict_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--result-json", required=True,
                        help="result.json from the training run")
    parser.add_argument("--merged-output", default=None,
                        help="optional ordinary FP32 state_dict .pt")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    with Path(args.result_json).open(encoding="utf-8") as handle:
        training_result = json.load(handle)
    config = training_result["configuration"]
    signature = training_result["checkpoint_signature"]
    with (Path(config["data_dir"]) / "manifest.json").open(encoding="utf-8") as handle:
        vocab_size = int(json.load(handle)["vocab_size"])
    merge_started = time.perf_counter()
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        args.checkpoint_dir,
        tag=args.tag,
        exclude_frozen_parameters=False,
    )
    merge_seconds = time.perf_counter() - merge_started
    model = build_model(
        block_size=int(config["block_size"]),
        vocab_size=vocab_size,
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        dropout=float(config["dropout"]),
        bias=bool(config["bias"]),
    )
    load_started = time.perf_counter()
    incompatible = model.load_state_dict(state_dict, strict=True)
    strict_load_seconds = time.perf_counter() - load_started
    model.eval()
    length = min(8, int(config["block_size"]))
    fixed_input = torch.arange(length, dtype=torch.long)[None, :] % vocab_size
    with torch.no_grad():
        logits, _ = model(fixed_input)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("ordinary GPT produced non-finite logits")
    if args.merged_output:
        atomic_torch_save(state_dict, args.merged_output)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "zero_stage": training_result["zero_stage"],
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_tag": args.tag,
        "checkpoint": directory_artifact_stats(
            Path(args.checkpoint_dir) / args.tag
        ),
        "merge_seconds": merge_seconds,
        "strict_load_seconds": strict_load_seconds,
        "peak_process_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "window_prefix_sha256": training_result["progress"]["global_window_prefix_sha256"],
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "tensor_count": len(state_dict),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "model_state_sha256": state_dict_sha256(state_dict),
        "fixed_input_logits_sha256": tensor_sha256(logits),
        "merged_output": args.merged_output,
        "merged_output_bytes": (
            Path(args.merged_output).stat().st_size
            if args.merged_output
            else None
        ),
        "signature": signature,
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

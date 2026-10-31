#!/usr/bin/env python3
"""Verify native contiguous-KV prefill/decode against full recomputation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import torch

from distributed_common import atomic_json_dump, sha256_file
from serving.artifact import load_inference_artifact
from serving.native import greedy_generate_cached, greedy_generate_naive
from serving.workloads import encode_prompts, prompt_set_sha256


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("results/g_serving/g2_kv_cache/correctness.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _compare_generation(
    naive: Any,
    cached: Any,
    *,
    atol: float,
) -> dict[str, Any]:
    if len(naive.step_logits) != len(cached.step_logits):
        raise ValueError("generation paths returned different step counts")
    step_differences = [
        float((left - right).abs().max())
        for left, right in zip(naive.step_logits, cached.step_logits)
    ]
    logits_close = all(
        torch.allclose(left, right, rtol=0.0, atol=atol)
        for left, right in zip(naive.step_logits, cached.step_logits)
    )
    token_ids_equal = torch.equal(naive.token_ids, cached.token_ids)
    mismatch = (naive.token_ids != cached.token_ids).flatten().nonzero()
    return {
        "status": "pass" if logits_close and token_ids_equal else "fail",
        "logits_allclose": bool(logits_close),
        "max_abs_difference_by_step": step_differences,
        "max_abs_difference": max(step_differences, default=0.0),
        "token_ids_equal": token_ids_equal,
        "first_token_mismatch": int(mismatch[0]) if mismatch.numel() else None,
        "generated_token_ids": naive.token_ids[:, -len(naive.step_logits) :].tolist(),
        "cache_rebuilds": cached.cache_rebuilds,
        "cache_allocated_bytes": cached.cache_allocated_bytes,
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    dtype_names = [name.strip() for name in args.dtypes.split(",") if name.strip()]
    unknown = sorted(set(dtype_names) - set(DTYPES))
    if unknown:
        raise ValueError(f"unsupported dtypes: {unknown}")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    device = _resolve_device(args.device)
    torch.set_num_threads(args.num_threads)
    prompts = encode_prompts()
    workloads: list[dict[str, Any]] = []

    for dtype_name in dtype_names:
        model, metadata = load_inference_artifact(
            args.artifact, device=device, dtype=DTYPES[dtype_name]
        )
        tolerance = 1e-5 if dtype_name == "float32" else 0.125
        for prompt_name, base_prompt in prompts.items():
            for batch_size in batch_sizes:
                prompt = base_prompt.expand(batch_size, -1).contiguous().to(device)
                started = time.perf_counter()
                naive = greedy_generate_naive(model, prompt, args.max_new_tokens)
                naive_seconds = time.perf_counter() - started
                started = time.perf_counter()
                cached = greedy_generate_cached(model, prompt, args.max_new_tokens)
                cached_seconds = time.perf_counter() - started
                comparison = _compare_generation(naive, cached, atol=tolerance)
                comparison.update(
                    {
                        "kind": "standard",
                        "dtype": dtype_name,
                        "prompt": prompt_name,
                        "prompt_tokens": int(prompt.shape[1]),
                        "batch_size": batch_size,
                        "max_new_tokens": args.max_new_tokens,
                        "naive_cpu_wall_seconds": naive_seconds,
                        "cached_cpu_wall_seconds": cached_seconds,
                        "cpu_wall_speedup": (
                            naive_seconds / cached_seconds if cached_seconds > 0 else None
                        ),
                    }
                )
                workloads.append(comparison)

        # Learned absolute positions require cache rebuild after a sliding-window
        # shift. Exercise that boundary once with the formal FP32 reference.
        if dtype_name == "float32":
            seed = prompts["long"]
            repeats = (model.config.block_size + seed.shape[1] - 1) // seed.shape[1]
            overflow_prompt = seed.repeat(1, repeats)[:, : model.config.block_size].to(device)
            naive = greedy_generate_naive(model, overflow_prompt, 2)
            cached = greedy_generate_cached(model, overflow_prompt, 2)
            comparison = _compare_generation(naive, cached, atol=tolerance)
            comparison.update(
                {
                    "kind": "block_boundary_rebuild",
                    "dtype": dtype_name,
                    "prompt": "exact_block_size",
                    "prompt_tokens": int(overflow_prompt.shape[1]),
                    "batch_size": 1,
                    "max_new_tokens": 2,
                    "expected_cache_rebuilds": 1,
                    "rebuild_count_correct": cached.cache_rebuilds == 1,
                }
            )
            if not comparison["rebuild_count_correct"]:
                comparison["status"] = "fail"
            workloads.append(comparison)

    passed = all(workload["status"] == "pass" for workload in workloads)
    result = {
        "schema_version": 1,
        "stage": "G2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "device": device,
        "torch_version": torch.__version__,
        "artifact": args.artifact.as_posix(),
        "artifact_sha256": sha256_file(args.artifact),
        "artifact_state_dict_sha256": metadata["state_dict_sha256"],
        "prompt_set_sha256": prompt_set_sha256(),
        "dtypes": dtype_names,
        "batch_sizes": batch_sizes,
        "workload_count": len(workloads),
        "workloads": workloads,
        "limitations": (
            [
                "GPU unavailable; CPU wall times are diagnostic and are not formal serving results",
                "Formal TTFT, decode throughput and peak CUDA memory remain pending",
            ]
            if device == "cpu"
            else []
        ),
    }
    atomic_json_dump(result, args.result_path)
    print(f"G2 KV-cache correctness: {result['status']} -> {args.result_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


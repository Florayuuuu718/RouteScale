#!/usr/bin/env python3
"""Locate the first greedy-token divergence between naive and cached decode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from distributed_common import atomic_json_dump, sha256_file
from serving.artifact import load_inference_artifact
from serving.native.benchmark import build_prompt_batch


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/tinystories_full/val.bin"))
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("results/g_serving/g3_native/pilot_divergence.json"),
    )
    return parser.parse_args()


@torch.inference_mode()
def diagnose(
    model,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    generated = prompt
    cache = model.allocate_kv_cache(batch_size=1)
    previous_token = None
    maximum_difference = 0.0
    per_step_maximum: list[float] = []
    common_tokens: list[int] = []
    first_divergence = None

    for step in range(max_new_tokens):
        context = generated[:, -model.config.block_size :]
        naive_logits, _ = model(context)
        naive_logits = naive_logits[:, -1, :].float()
        if step == 0:
            cached_logits = model.forward_with_kv_cache(context, cache)[:, -1, :].float()
        elif cache.length == cache.capacity:
            cache = model.allocate_kv_cache(batch_size=1)
            cached_logits = model.forward_with_kv_cache(context, cache)[:, -1, :].float()
        else:
            cached_logits = model.forward_with_kv_cache(previous_token, cache)[:, -1, :].float()

        difference = (naive_logits - cached_logits).abs()
        step_maximum = float(difference.max())
        maximum_difference = max(maximum_difference, step_maximum)
        per_step_maximum.append(step_maximum)
        naive_token = torch.argmax(naive_logits, dim=-1, keepdim=True)
        cached_token = torch.argmax(cached_logits, dim=-1, keepdim=True)
        if not torch.equal(naive_token, cached_token):
            naive_values, naive_indices = torch.topk(naive_logits, k=5, dim=-1)
            cached_values, cached_indices = torch.topk(cached_logits, k=5, dim=-1)
            candidate_ids = sorted(
                {int(value) for value in naive_indices.flatten()}
                | {int(value) for value in cached_indices.flatten()}
            )
            first_divergence = {
                "step": step,
                "naive_token_id": int(naive_token),
                "cached_token_id": int(cached_token),
                "naive_top5_token_ids": naive_indices.flatten().tolist(),
                "naive_top5_logits": naive_values.flatten().tolist(),
                "cached_top5_token_ids": cached_indices.flatten().tolist(),
                "cached_top5_logits": cached_values.flatten().tolist(),
                "naive_top1_top2_margin": float(naive_values[0, 0] - naive_values[0, 1]),
                "cached_top1_top2_margin": float(cached_values[0, 0] - cached_values[0, 1]),
                "candidate_logits": {
                    str(token_id): {
                        "naive": float(naive_logits[0, token_id]),
                        "cached": float(cached_logits[0, token_id]),
                        "absolute_difference": float(difference[0, token_id]),
                    }
                    for token_id in candidate_ids
                },
                "step_max_abs_difference": step_maximum,
            }
            break
        common_tokens.append(int(naive_token))
        previous_token = naive_token
        generated = torch.cat((generated, naive_token), dim=1)

    return {
        "checked_steps": len(per_step_maximum),
        "common_greedy_tokens": common_tokens,
        "first_divergence": first_divergence,
        "max_abs_difference": maximum_difference,
        "max_abs_difference_by_step": per_step_maximum,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype_names = [name.strip() for name in args.dtypes.split(",") if name.strip()]
    unknown = sorted(set(dtype_names) - set(DTYPES))
    if unknown:
        raise ValueError(f"unsupported dtypes: {unknown}")
    prompt_cpu, workload = build_prompt_batch(
        args.data_path,
        batch_size=1,
        prompt_length=args.prompt_length,
        prefix_mode="none",
        seed=args.seed,
    )
    prompt = prompt_cpu.to(args.device)
    diagnostics: dict[str, Any] = {}
    for dtype_name in dtype_names:
        model, _ = load_inference_artifact(
            args.artifact, device=args.device, dtype=DTYPES[dtype_name]
        )
        diagnostics[dtype_name] = diagnose(
            model, prompt, max_new_tokens=args.max_new_tokens
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "stage": "G3",
        "diagnostic": "naive_kv_greedy_divergence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "torch_version": torch.__version__,
        "artifact": args.artifact.as_posix(),
        "artifact_sha256": sha256_file(args.artifact),
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "workload": workload,
        "dtypes": diagnostics,
    }
    atomic_json_dump(result, args.result_path)
    for dtype_name, diagnostic in diagnostics.items():
        divergence = diagnostic["first_divergence"]
        message = "none" if divergence is None else f"step {divergence['step']}"
        print(
            f"{dtype_name}: first divergence={message}, "
            f"max abs diff={diagnostic['max_abs_difference']:.6g}"
        )
    print(f"wrote {args.result_path}")


if __name__ == "__main__":
    main()


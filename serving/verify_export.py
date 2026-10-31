#!/usr/bin/env python3
"""Verify source/export logits and greedy decoding in a fresh process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import time
from typing import Any

import torch
from distributed_common import atomic_json_dump, atomic_torch_save, sha256_file
from serving.artifact import load_inference_artifact, load_training_checkpoint_model
from serving.workloads import G1_PROMPTS, encode_prompts, prompt_set_sha256
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("runs/checkpoints/g_serving/dense_baseline/reference_logits.pt"),
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("results/g_serving/g1_export/equivalence.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--greedy-tokens", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


@torch.inference_mode()
def _capture(
    model: torch.nn.Module,
    prompts: dict[str, torch.Tensor],
    *,
    device: str,
    greedy_tokens: int,
) -> dict[str, dict[str, Any]]:
    captured: dict[str, dict[str, Any]] = {}
    for name, cpu_ids in prompts.items():
        ids = cpu_ids.to(device)
        logits, _ = model(ids)
        generated = ids
        for _ in range(greedy_tokens):
            context = generated[:, -model.config.block_size :]
            step_logits, _ = model(context)
            next_token = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        captured[name] = {
            "logits": logits[:, -1, :].float().cpu(),
            "greedy_tokens": generated[:, -greedy_tokens:].cpu(),
        }
    return captured


def _first_mismatch(left: torch.Tensor, right: torch.Tensor) -> int | None:
    unequal = (left != right).flatten().nonzero()
    return int(unequal[0]) if unequal.numel() else None


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.greedy_tokens <= 0:
        raise ValueError("greedy-tokens must be positive")
    requested_dtypes = [name.strip() for name in args.dtypes.split(",") if name.strip()]
    unknown = sorted(set(requested_dtypes) - set(DTYPES))
    if unknown:
        raise ValueError(f"unsupported dtypes: {unknown}")
    device = _resolve_device(args.device)
    torch.set_num_threads(args.num_threads)
    prompts = encode_prompts()
    references: dict[str, Any] = {
        "prompt_token_ids": prompts,
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "dtypes": {},
    }
    comparisons: dict[str, Any] = {}
    strict_loads = 0
    started = time.perf_counter()

    for dtype_name in requested_dtypes:
        dtype = DTYPES[dtype_name]
        source_model, source_config, _ = load_training_checkpoint_model(
            args.checkpoint, device=device, dtype=dtype
        )
        source_capture = _capture(
            source_model, prompts, device=device, greedy_tokens=args.greedy_tokens
        )
        del source_model
        gc.collect()

        exported_model, artifact_metadata = load_inference_artifact(
            args.artifact, device=device, dtype=dtype
        )
        strict_loads += 1
        exported_capture = _capture(
            exported_model, prompts, device=device, greedy_tokens=args.greedy_tokens
        )
        tied_storage = (
            exported_model.transformer.wte.weight.data_ptr()
            == exported_model.lm_head.weight.data_ptr()
        )
        del exported_model
        gc.collect()

        dtype_references: dict[str, Any] = {}
        dtype_comparisons: dict[str, Any] = {}
        atol = 1e-6 if dtype_name == "float32" else 1e-2
        for prompt_name in prompts:
            source_logits = source_capture[prompt_name]["logits"]
            exported_logits = exported_capture[prompt_name]["logits"]
            source_tokens = source_capture[prompt_name]["greedy_tokens"]
            exported_tokens = exported_capture[prompt_name]["greedy_tokens"]
            difference = (source_logits - exported_logits).abs()
            greedy_equal = torch.equal(source_tokens, exported_tokens)
            logits_close = torch.allclose(
                source_logits, exported_logits, atol=atol, rtol=0.0
            )
            dtype_references[prompt_name] = {
                "logits": source_logits,
                "greedy_tokens": source_tokens,
            }
            top_values, top_indices = torch.topk(source_logits, k=5, dim=-1)
            dtype_comparisons[prompt_name] = {
                "prompt_tokens": int(prompts[prompt_name].numel()),
                "logits_allclose": bool(logits_close),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
                "source_logits_sha256": _tensor_sha256(source_logits),
                "exported_logits_sha256": _tensor_sha256(exported_logits),
                "greedy_tokens_equal": greedy_equal,
                "first_greedy_mismatch": _first_mismatch(source_tokens, exported_tokens),
                "greedy_token_ids": source_tokens.flatten().tolist(),
                "top5_token_ids": top_indices.flatten().tolist(),
                "top5_logits": top_values.flatten().tolist(),
            }
        references["dtypes"][dtype_name] = dtype_references
        comparisons[dtype_name] = {
            "source_model_config": source_config.__dict__,
            "artifact_state_dict_sha256": artifact_metadata["state_dict_sha256"],
            "strict_load": True,
            "tied_embedding_storage": tied_storage,
            "prompts": dtype_comparisons,
        }

    atomic_torch_save(references, args.reference_path)
    all_prompt_checks = [
        prompt_check
        for dtype_check in comparisons.values()
        for prompt_check in dtype_check["prompts"].values()
    ]
    passed = (
        strict_loads == len(requested_dtypes)
        and all(item["strict_load"] and item["tied_embedding_storage"]
                for item in comparisons.values())
        and all(item["logits_allclose"] and item["greedy_tokens_equal"]
                for item in all_prompt_checks)
    )
    result = {
        "schema_version": 1,
        "stage": "G1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "device": device,
        "torch_version": torch.__version__,
        "dtypes": requested_dtypes,
        "greedy_tokens": args.greedy_tokens,
        "prompt_set_sha256": prompt_set_sha256(G1_PROMPTS),
        "source_checkpoint": args.checkpoint.as_posix(),
        "source_checkpoint_sha256": references["source_checkpoint_sha256"],
        "artifact": args.artifact.as_posix(),
        "artifact_sha256": sha256_file(args.artifact),
        "reference_logits": args.reference_path.as_posix(),
        "reference_logits_sha256": sha256_file(args.reference_path),
        "strict_load_count": strict_loads,
        "elapsed_seconds": time.perf_counter() - started,
        "comparisons": comparisons,
        "limitations": (
            ["GPU unavailable; this run establishes correctness only, not serving performance"]
            if device == "cpu"
            else []
        ),
    }
    atomic_json_dump(result, args.result_path)
    print(f"G1 export equivalence: {result['status']} -> {args.result_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


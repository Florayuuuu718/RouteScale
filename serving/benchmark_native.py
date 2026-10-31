#!/usr/bin/env python3
"""Run one isolated G3 native serving benchmark configuration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
from typing import Any

import torch

from distributed_common import atomic_json_dump, git_metadata, sha256_file
from serving.artifact import load_inference_artifact
from serving.native.benchmark import (
    BACKENDS,
    PREFIX_MODES,
    build_decode_batch,
    build_prompt_batch,
    generate_timed,
    summarize_measurements,
)


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/tinystories_full/val.bin"))
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--prefix-mode", choices=sorted(PREFIX_MODES), default="none")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument(
        "--decode-input-policy",
        choices=("fixed", "greedy"),
        default="fixed",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--measure-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="allow schema/logic smoke only; result is explicitly non-formal",
    )
    return parser.parse_args()


def _environment(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": device,
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal G3 benchmark requires an accessible CUDA device")
    if (
        args.device == "cuda"
        and args.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("formal BF16 G3 benchmark requires BF16-capable CUDA hardware")
    if args.device == "cpu" and not args.allow_cpu_smoke:
        raise RuntimeError("CPU is permitted only with --allow-cpu-smoke")
    if min(
        args.batch_size,
        args.prompt_length,
        args.output_length,
        args.warmup_repeats,
        args.measure_repeats,
    ) <= 0:
        raise ValueError("batch, sequence and repeat counts must be positive")

    model, artifact_metadata = load_inference_artifact(
        args.artifact, device=args.device, dtype=DTYPES[args.dtype]
    )
    if args.prompt_length > model.config.block_size:
        raise ValueError("prompt length exceeds model block_size")
    prompt_cpu, workload = build_prompt_batch(
        args.data_path,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        prefix_mode=args.prefix_mode,
        seed=args.seed,
    )
    prompt = prompt_cpu.to(args.device)
    forced_tokens = None
    if args.decode_input_policy == "fixed":
        forced_cpu, decode_workload = build_decode_batch(
            args.data_path,
            batch_size=args.batch_size,
            output_length=args.output_length,
            seed=args.seed + 1,
        )
        forced_tokens = forced_cpu.to(args.device)
        workload.update(decode_workload)
    else:
        workload.update(
            {
                "decode_input_policy": "greedy",
                "decode_seed": None,
                "decode_token_ids_sha256": None,
            }
        )

    for _ in range(args.warmup_repeats):
        generate_timed(
            model,
            prompt,
            output_length=args.output_length,
            backend=args.backend,
            forced_token_ids=forced_tokens,
        )
    if args.device == "cuda":
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)

    measurements = [
        generate_timed(
            model,
            prompt,
            output_length=args.output_length,
            backend=args.backend,
            forced_token_ids=forced_tokens,
        )
        for _ in range(args.measure_repeats)
    ]
    metrics = summarize_measurements(
        measurements,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        output_length=args.output_length,
    )
    if args.device == "cuda":
        metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        metrics["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
    else:
        metrics["peak_allocated_bytes"] = None
        metrics["peak_reserved_bytes"] = None

    result = {
        "schema_version": 1,
        "stage": "G3",
        "benchmark": "native_serving",
        "status": "pass",
        "evidence_class": "formal_gpu" if args.device == "cuda" else "cpu_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "version": {
            "git": git_metadata(),
            "source_sha256": {
                path: sha256_file(path)
                for path in (
                    "model.py",
                    "serving/artifact.py",
                    "serving/native/benchmark.py",
                    "serving/benchmark_native.py",
                )
            },
        },
        "environment": _environment(args.device),
        "artifact": {
            "path": args.artifact.as_posix(),
            "sha256": sha256_file(args.artifact),
            "state_dict_sha256": artifact_metadata["state_dict_sha256"],
        },
        "workload": workload,
        "configuration": {
            "backend": args.backend,
            "prefix_mode": args.prefix_mode,
            "batch_size": args.batch_size,
            "prompt_length": args.prompt_length,
            "output_length": args.output_length,
            "decode_input_policy": args.decode_input_policy,
            "dtype": args.dtype,
            "warmup_repeats": args.warmup_repeats,
            "measure_repeats": args.measure_repeats,
            "block_size": model.config.block_size,
            "context_overflow": args.prompt_length + args.output_length > model.config.block_size,
        },
        "measurement": {
            "timing": (
                "per-token CUDA Events plus synchronized request wall time"
                if args.device == "cuda"
                else "perf_counter CPU smoke; not formal performance evidence"
            ),
            **metrics,
        },
        "limitations": (
            []
            if args.device == "cuda"
            else ["CPU smoke validates the harness only; performance values are non-formal"]
        ),
    }
    atomic_json_dump(result, args.result_path)
    print(
        f"G3 {result['evidence_class']} {args.backend}: "
        f"TTFT p50={metrics['ttft_p50_ms']:.3f} ms, "
        f"output={metrics['output_tokens_per_second']:.1f} token/s -> "
        f"{args.result_path}"
    )


if __name__ == "__main__":
    main()


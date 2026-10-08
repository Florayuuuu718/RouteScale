"""Benchmark the two formal MoE expert GEMM shapes in isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
import triton

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_grouped_gemm import grouped_gemm  # noqa: E402


FORMAL_SHAPES = (
    (4, 1280, 512, 2048),
    (4, 1280, 2048, 512),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def benchmark_shape(
    shape: tuple[int, int, int, int],
    *,
    warmup_ms: int,
    repetition_ms: int,
) -> dict:
    groups, rows, reduction, columns = shape
    left = torch.randn(
        groups,
        rows,
        reduction,
        device="cuda",
        dtype=torch.bfloat16,
    )
    right = torch.randn(
        groups,
        reduction,
        columns,
        device="cuda",
        dtype=torch.bfloat16,
    )
    functions = {
        "python_loop": lambda: torch.stack(
            [left[index] @ right[index] for index in range(groups)]
        ),
        "torch_bmm": lambda: torch.bmm(left, right),
        "triton_grouped": lambda: grouped_gemm(left, right),
    }
    reference = functions["torch_bmm"]()
    actual = functions["triton_grouped"]()
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-1)

    flop_count = 2 * groups * rows * reduction * columns
    methods = {}
    for name, function in functions.items():
        median_ms, low_ms, high_ms = triton.testing.do_bench(
            function,
            warmup=warmup_ms,
            rep=repetition_ms,
            quantiles=[0.5, 0.2, 0.8],
        )
        methods[name] = {
            "median_ms": median_ms,
            "p20_ms": low_ms,
            "p80_ms": high_ms,
            "tflops": flop_count / median_ms / 1e9,
        }
    methods["triton_grouped"]["max_absolute_error"] = (
        actual.float() - reference.float()
    ).abs().max().item()
    return {
        "groups": groups,
        "M": rows,
        "N": columns,
        "K": reduction,
        "dtype": "bfloat16",
        "methods": methods,
        "triton_vs_python_loop_speedup": (
            methods["python_loop"]["median_ms"]
            / methods["triton_grouped"]["median_ms"]
        ),
        "triton_vs_torch_bmm_speedup": (
            methods["torch_bmm"]["median_ms"]
            / methods["triton_grouped"]["median_ms"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e_triton_moe/grouped_gemm.json"),
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repetition-ms", type=int, default=300)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E-stage Grouped GEMM benchmark requires CUDA")

    source_paths = [
        Path("triton_grouped_gemm.py"),
        Path("scripts/benchmark_e_grouped_gemm.py"),
    ]
    result = {
        "schema_version": 1,
        "experiment": "E-stage fixed-capacity Grouped GEMM microbenchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "triton": triton.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "sm_count": torch.cuda.get_device_properties("cuda").multi_processor_count,
            "driver": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
        "protocol": {
            "warmup_ms": args.warmup_ms,
            "repetition_ms": args.repetition_ms,
            "timing": "triton.testing.do_bench",
            "scope": "forward GEMM only; excludes routing, packing, activation, and backward",
        },
        "source_sha256": {
            str(path): sha256_file(path) for path in source_paths
        },
        "shapes": [
            benchmark_shape(
                shape,
                warmup_ms=args.warmup_ms,
                repetition_ms=args.repetition_ms,
            )
            for shape in FORMAL_SHAPES
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["shapes"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

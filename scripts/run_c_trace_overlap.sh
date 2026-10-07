#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" scripts/summarize_c_trace_overlap.py \
  --repo-root=. \
  --manifest=results/c_ddp_autodl/profiler/4gpu_autodl_manifest.json \
  --manifest=results/c_backend_profiler/deepspeed_zero0/4gpu_autodl_manifest.json \
  --manifest=results/c_backend_profiler/deepspeed_zero1/4gpu_autodl_manifest.json \
  --manifest=results/c_backend_profiler/deepspeed_zero2/4gpu_autodl_manifest.json \
  --manifest=results/c_backend_profiler/deepspeed_zero3/4gpu_autodl_manifest.json \
  --manifest=results/c_backend_profiler/fsdp2/4gpu_autodl_manifest.json \
  --output=results/c_trace_overlap/summary.json

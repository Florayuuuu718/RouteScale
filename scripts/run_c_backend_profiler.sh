#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
RESULTS_DIR=${RESULTS_DIR:-results/c_backend_profiler}
RUN_ID=${RUN_ID:-4gpu_autodl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

for stage in 0 1 2 3; do
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    train_deepspeed.py \
    --deepspeed-config="config/deepspeed/zero${stage}.json" \
    --data-dir=data/tinystories_full \
    --out-dir="$RESULTS_DIR/runtime/deepspeed_zero${stage}" \
    --tail-policy=drop \
    --fused-optimizer \
    --profiler \
    --profiler-run-id="$RUN_ID" \
    --profiler-results-dir="$RESULTS_DIR" \
    --profiler-trace-dir="$RESULTS_DIR/traces"
done

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_fsdp2.py \
  --data-dir=data/tinystories_full \
  --out-dir="$RESULTS_DIR/runtime/fsdp2" \
  --tail-policy=drop \
  --fused-optimizer \
  --profiler \
  --profiler-run-id="$RUN_ID" \
  --profiler-results-dir="$RESULTS_DIR" \
  --profiler-trace-dir="$RESULTS_DIR/traces"

"$PYTHON_BIN" scripts/summarize_c_backend_profiler.py \
  --results-dir="$RESULTS_DIR" \
  --run-id="$RUN_ID" \
  --expected-ranks=4 \
  --variants \
    deepspeed_zero0 \
    deepspeed_zero1 \
    deepspeed_zero2 \
    deepspeed_zero3 \
    fsdp2

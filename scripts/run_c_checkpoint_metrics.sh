#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
RESULTS_DIR=${RESULTS_DIR:-results/c_checkpoint_metrics}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

for stage in 2 3; do
  save_dir="$RESULTS_DIR/deepspeed_zero${stage}_save"
  resume_dir="$RESULTS_DIR/deepspeed_zero${stage}_resume"

  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    train_deepspeed.py \
    --deepspeed-config="config/deepspeed/zero${stage}.json" \
    --data-dir=data/tinystories_debug \
    --out-dir="$save_dir" \
    --fused-optimizer \
    --max-updates=2

  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    train_deepspeed.py \
    --deepspeed-config="config/deepspeed/zero${stage}.json" \
    --data-dir=data/tinystories_debug \
    --out-dir="$resume_dir" \
    --resume-dir="$save_dir/checkpoints" \
    --resume-tag=update-00000002 \
    --fused-optimizer \
    --max-updates=3

  "$PYTHON_BIN" scripts/verify_deepspeed_fp32.py \
    --checkpoint-dir="$save_dir/checkpoints" \
    --tag=update-00000002 \
    --result-json="$save_dir/result.json" \
    --merged-output="$save_dir/merged_fp32.pt" \
    --output="$save_dir/fp32_verification.json"
done

fsdp_save="$RESULTS_DIR/fsdp2_save"
fsdp_resume="$RESULTS_DIR/fsdp2_resume"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_fsdp2.py \
  --data-dir=data/tinystories_debug \
  --out-dir="$fsdp_save" \
  --fused-optimizer \
  --max-updates=2

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_fsdp2.py \
  --data-dir=data/tinystories_debug \
  --out-dir="$fsdp_resume" \
  --resume-from="$fsdp_save/checkpoints/update-00000002" \
  --fused-optimizer \
  --max-updates=3

"$PYTHON_BIN" scripts/summarize_c_checkpoints.py \
  --results-dir="$RESULTS_DIR"

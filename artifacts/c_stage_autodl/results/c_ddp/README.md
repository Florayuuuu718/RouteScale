# C-stage DDP prerequisites and local single-GPU validation

Date: 2026-10-06. This directory records the parts of C that can be validated
on the current one-GPU machine. It is not a substitute for the same-machine
1/2/4-GPU measurements required for scaling claims.

## Completed locally

- `ddp_windows.py` creates one deterministic global sequence of non-overlapping
  windows and maps it to rank, micro-step, and batch slot.
- CPU tests cover 1/2/4 logical ranks, deterministic reconstruction, no
  duplicate/omitted windows within complete updates, and explicit `drop`/`pad`
  tail accounting.
- `scripts/check_ddp_update.py` compares one sharded update with an equivalent
  full-global-batch update. The one-process CUDA/NCCL check passed exactly. A
  two-process CPU/Gloo check also passed, with `3.725e-09` maximum parameter
  error and zero difference between replicas. The CPU result validates DDP
  averaging and the harness, but not multi-GPU NCCL numerics.
- The no-Profiler C benchmark records every rank's CUDA Event timings, window
  hash, loss, and peak memory. Rank 0 derives the slowest-rank step time and
  global throughput.

## Local one-GPU strong-scaling anchor

Configuration: `config/benchmark_ddp_tinystories.py`, `compile=False`, BF16,
micro-batch 8, four accumulation steps, and 16,384 tokens/update. Each of three
independent processes used 20 warmup and 100 measured updates.

| Run | Median step | Tokens/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| 1 | 346.696 ms | 47,258 | 3,906.9 MiB | 4,910.0 MiB |
| 2 | 346.479 ms | 47,287 | 3,906.9 MiB | 4,910.0 MiB |
| 3 | 346.310 ms | 47,310 | 3,906.9 MiB | 4,910.0 MiB |

The median of run medians is **346.479 ms/update**, or **47,287 tokens/s**.
Every run consumed the same 3,200 measured windows, reported no repeated
measurement window, matched its planned window hash
(`aa0764b0...c82c885e`), and kept all recorded losses finite.

These local results validate the C benchmark implementation. The target
four-GPU server must repeat the 1-GPU measurement; only that same-server value
may be used as the denominator for 2/4-GPU speedup and efficiency.

## Reproduction

Run the local correctness checks:

```bash
uv run torchrun --standalone --nproc_per_node=1 \
  scripts/check_ddp_update.py --device cuda
uv run torchrun --standalone --nproc_per_node=2 \
  scripts/check_ddp_update.py --device cpu
```

Run the one-GPU benchmark three times and validate it:

```bash
for run_id in run1 run2 run3; do
  uv run python train.py config/benchmark_ddp_tinystories.py \
    --ddp_run_id="$run_id"
done
uv run python scripts/summarize_c_ddp.py \
  --results-dir results/c_ddp/strong --world-sizes 1
```

On the target server, strong scaling uses the same configuration with
`torchrun --nproc_per_node=2` and `4`. Weak scaling adds
`--ddp_scaling_mode="weak"`. Run each GPU count in a fresh process three times,
then summarize without `--world-sizes` so all available GPU counts are checked.

## Still requires the multi-GPU server

- two-GPU CUDA/NCCL and four-GPU update equivalence;
- same-server 1/2/4-GPU strong and weak scaling;
- per-rank NCCL/AllReduce traces and topology analysis;
- recoverable full-window-coverage training.

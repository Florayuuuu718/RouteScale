# RouteScale experiment records

## A2: local TinyStories candidate

Date: 2026-09-20. These runs validate the local training configuration on one
NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB). They are not formal throughput
benchmarks because CUDA timing boundaries have not yet been synchronized.

Data: `tinystories_full`, revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

- `train.bin`: 473,992,236 tokens, SHA256
  `66c5c49a38ebcab576854f2f58fad55e06b2bb29ecbc34fa421910e4beb37500`
- `val.bin`: 4,765,918 tokens, SHA256
  `f0d47c000fdf2f7fbae1159832002b079a338cfde9ce4c5f65413da68386f5be`

Candidate model: 8 Transformer Blocks, 8 attention heads, embedding width 512,
sequence length 512, BF16 autocast, `compile=False`, gradient accumulation 4,
50.91M non-position-embedding parameters.

### Micro-batch fit check

Each run performed 20 optimizer updates. Step times are diagnostic only.

| Micro-batch | Tokens/update | Peak allocated | Peak reserved | Typical logged step |
|---:|---:|---:|---:|---:|
| 4 | 8,192 | 2426.3 MiB | 2772.0 MiB | ~184 ms |
| 6 | 12,288 | 3157.7 MiB | 3968.0 MiB | ~264 ms |
| 8 | 16,384 | 3899.2 MiB | 5148.0 MiB | ~346 ms |

Micro-batch 8 was selected. Its peak reserved memory was about 63% of device
memory, leaving headroom for evaluation, profiling, and runtime variation.

### Stability run

Configuration: `config/train_tinystories.py`; exactly 300 optimizer updates.

- Validation loss estimate (20 random batches): 10.9094 at step 0, 4.0805 at
  step 100, 3.7706 at step 200, and 3.5545 at step 300.
- No NaN or CUDA OOM occurred.
- Peak CUDA memory: 3899.2 MiB allocated, 5148.0 MiB reserved.
- Checkpoint: `out-tinystories/ckpt.pt`, 585.6 MiB, `iter_num=300`.
- Loading the checkpoint and continuing with updates 301 and 302 succeeded.

The fixed `max_iters` boundary now treats `max_iters` as the exact number of
completed optimizer updates. Evaluation and checkpoint I/O are excluded from
the following iteration's CPU wall-time measurement, but B-stage benchmarks
still need explicit CUDA synchronization or CUDA events.

## B1: reproducible single-GPU baseline without Profiler

Date: 2026-10-01. Configuration: `config/benchmark_tinystories.py`. The model,
data, precision and workload are unchanged from the A2 candidate: 8 blocks, 8
heads, width 512, sequence length 512, BF16, `compile=False`, micro-batch 8 and
four gradient-accumulation micro-steps. Each optimizer update processes 16,384
training positions.

The benchmark uses a dedicated data RNG and records the SHA256 of every window
offset consumed in the measured interval. Each independent process starts from
the same model seed, warms up for 20 optimizer updates, then measures 100
updates with per-update CUDA Events. Evaluation, checkpoints, ordinary logging
and PyTorch Profiler are disabled in the measured interval.

Command, repeated with `run1`, `run2` and `run3`:

```bash
uv run python train.py config/benchmark_tinystories.py \
  --benchmark_run_id="run1"
```

Raw results are in `results/b1_single_gpu/run1.json` through `run3.json`; the
validated aggregate is `results/b1_single_gpu/summary.json`.

| Run | Median step | Tokens/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| 1 | 347.097 ms | 47,203 | 3,906.9 MiB | 4,910.0 MiB |
| 2 | 346.388 ms | 47,300 | 3,906.9 MiB | 4,910.0 MiB |
| 3 | 346.390 ms | 47,299 | 3,906.9 MiB | 4,910.0 MiB |

The median of the three run medians is **346.390 ms/update**, or **47,299
tokens/s**. All three runs consumed the same 3,200 measured windows
(`b920fdb1...91157`), all measured losses were finite, and the last observed
micro-batch loss was approximately 3.92. The run-median range is 0.709 ms
(about 0.20% of the reported median), so this baseline is stable enough for
the B2 no-Profiler before/after comparisons.

The measurements were made from an uncommitted working tree while developing
the B1 harness. Each raw JSON therefore records the current HEAD, Git status,
tracked diff hash and hashes of the source/config files used. A future formal
cross-machine run should use a committed revision.

## B2: PyTorch Profiler and `torch.compile` A/B

Date: 2026-10-01. The frozen B1 workload was profiled after 20 unprofiled
startup updates with a `wait=2, warmup=2, active=10` PyTorch Profiler schedule.
The trace records CPU and CUDA activities, shapes and memory, with explicit
`get_batch`, `forward`, `backward`, `gradient_clipping`, `optimizer_step` and
`zero_grad` ranges. Profiler throughput is not reported as a formal result.

### Baseline trace finding

In the eager (`compile=False`) trace, matrix multiplication was the dominant
CUDA operator: `aten::mm` accumulated 2,184.5 ms of self-device time across the
10 active updates and accounted for 61.63% of the table's reported self CUDA
time. The same trace contained:

- 4,880 `aten::copy_` calls, totaling 396.7 ms of self-device time;
- 17,460 combined `cudaLaunchKernel` and `cuLaunchKernel` calls;
- only 0.324 ms of pinned H2D copies across the 10 active updates;
- 44.0 ms in the fused AdamW range and 18.3 ms in gradient clipping across the
  10 active updates.

The evidence does not support optimizing the data transfer path first. The
large number of launches and copy/pointwise operations motivated one isolated
change: enable `torch.compile` while keeping the model, data windows, batch,
precision, optimizer and measured work fixed.

### No-Profiler A/B

Each side used three independent processes, 20 warmup updates and 100 measured
updates per process. All six runs consumed the same measured-window sequence
(`b920fdb1...91157`) and all observed losses were finite.

| Mode | Run medians (ms/update) | Median | Tokens/s | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|---:|
| Eager | 346.810, 346.381, 346.273 | 346.381 ms | 47,301 | 3,906.9 MiB | 4,910.0 MiB |
| `torch.compile` | 239.856, 240.253, 239.897 | 239.897 ms | 68,296 | 3,128.6 MiB | 3,382.0 MiB |

For this fixed shape and hardware, `torch.compile` produced:

- **1.444x end-to-end speedup**;
- **30.74% lower step time**;
- **44.39% higher tokens/s**;
- 778.4 MiB lower peak allocated memory and 1,528.0 MiB lower peak reserved
  memory in this measurement.

The compiled trace supports the mechanism: Profiler self CUDA time was 29.6%
lower, combined launch API calls were 20.4% lower, `aten::copy_` calls fell from
4,880 to 160, and `aten::mm` self-device time was 31.8% lower. H2D time remained
negligible. Profiler event totals contain instrumentation overhead and overlap
across abstraction levels; these percentages explain the trace but do not
replace the no-Profiler result above.

Raw summaries are in `results/b2_profiler/` and `results/b2_compile_ab/`. The
large Chrome traces remain under ignored `logs/b2_profiler/` (109 MiB eager and
55 MiB compiled) and are not committed.

### Limits

Compilation/startup time is excluded from the steady-state interval, so short
jobs may not recover that cost. The result applies to the tested fixed shapes,
PyTorch 2.12.1, CUDA 12.9 and RTX 5060 Laptop GPU; the target four-GPU server
must be measured again. Finite and closely matched losses establish a short-run
correctness check, not bitwise equivalence or long-training quality parity.

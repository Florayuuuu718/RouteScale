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

## C: deterministic DDP prerequisites and local one-GPU benchmark

Date: 2026-10-06. The global-window scheduler, 1/2/4 logical-rank CPU tests,
single-update correctness harness, per-rank benchmark schema, and scaling
summary script are implemented. A two-process CPU/Gloo update matched the
equivalent full-batch model within `3.725e-09` maximum parameter error, with
identical parameters across ranks. The one-process CUDA/NCCL comparison was
exact.

The C no-Profiler benchmark was then run in three independent one-GPU
processes, each with 20 warmup and 100 measured updates. Its run medians were
346.696, 346.479 and 346.310 ms/update. The median of medians is **346.479
ms/update**, or **47,287 tokens/s**, with 3,906.9 MiB peak allocated and 4,910.0
MiB peak reserved memory. All losses were finite, runtime window hashes matched
the plan, and each run used the same 3,200 unique measured windows.

Full commands, raw JSON and scope limits are in
[`c_ddp/README.md`](c_ddp/README.md). These laptop results validated the harness
only and were not used as the denominator for the completed target-server
2/4-GPU scaling experiment.

## C5: deterministic coverage and checkpoint recovery

The 50.91M model completed one deterministic pass over every non-overlapping
window in `tinystories_debug`: 4,222 unique windows, 4,224 scheduled positions,
two explicitly padded repeats, and 413 trailing training tokens outside a full
512-token window. Coverage was 100%. The run performed 132 optimizer updates
in 55.41 seconds including periodic checkpoint work and reached 3,899.2 MiB
peak allocated memory. This elapsed time is an operational coverage duration,
not a no-checkpoint throughput benchmark.

Full-validation loss, weighted over all 194,558 target tokens including the
short final segment, changed from 10.9056 to 3.9285. An independent recovery
check compared `2 updates + save + new-process resume to 4` against four
uninterrupted updates. Model SHA256, loss history, next update and consumed
window hash all matched. See [`c5_coverage/README.md`](c5_coverage/README.md).

## C6/C7: local DeepSpeed and FSDP2 scaffolding (historical precursor)

DeepSpeed 0.19.7 ZeRO stages 0/1/2/3 each completed a one-GPU update,
sharded-format save, and new-process recovery. ZeRO-0 and native PyTorch used
the same four-step loss sequence; their final parameters differed by at most
`7.451e-09`, below the declared `1e-7` tolerance. FSDP2 block-level
`fully_shard` completed the same two-step save/resume-to-four workflow using
Distributed Checkpoint. ZeRO-2 and ZeRO-3 checkpoints were also consolidated
to ordinary FP32 state dictionaries and strictly loaded by the non-DeepSpeed
`GPT`, with no missing or unexpected keys.

These one-rank runs validate engine APIs, accumulation boundaries and state
lifecycle. They do not demonstrate state partitioning, collective cost or
memory savings, which require the target four-GPU process group. The tiny
smoke timings are deliberately not treated as performance results. Details are
in [`c6_deepspeed/README.md`](c6_deepspeed/README.md),
[`c7_fsdp2/README.md`](c7_fsdp2/README.md), and
[`c_distributed_smoke/trajectory_comparison.json`](c_distributed_smoke/trajectory_comparison.json).

## C target-server completion

The four-GPU target-server campaign is complete. The formal 51.17M-parameter
matrix covered native DDP, DeepSpeed ZeRO-0/1/2/3 and FSDP2 at 1/2/4 GPUs,
with three independent no-Profiler runs per point. DDP strong scaling reached
320,468 tokens/s on four GPUs (1.694x, 42.3% efficiency); weak scaling reached
570,219 tokens/s (3.014x, 75.3% efficiency). ZeRO-0 delivered the highest
four-GPU small-model throughput at 357,031 tokens/s, while native PyTorch was
the fastest single-GPU path.

Capacity probes bracketed native DDP at 694.2M success / 984.1M OOM,
ZeRO-2 at 2.041B success / 2.156B OOM, and ZeRO-3 at 2.274B success / 2.395B
OOM. FSDP2 successfully trained the 984.1M DDP-OOM configuration; its upper
failure boundary was not searched. Four-GPU checkpoint save/resume was
verified for ZeRO-2, ZeRO-3 and FSDP2, and ZeRO-2/3 were consolidated to
ordinary FP32 state dictionaries and strictly loaded by the native model.
At the same reopened-instance topology and 984.1M configuration, ZeRO-3
reached 33,805 tokens/s with 14.020/17.436 GiB allocated/reserved, while
FSDP2 reached 22,481 tokens/s with 12.091/13.898 GiB. This is a throughput
versus GPU-memory tradeoff, not a universal winner.

The committed, hash-verified server evidence lives under
[`../artifacts/c_stage_autodl/`](../artifacts/c_stage_autodl/), rather than
being duplicated into this historical local-results tree. See
[`../docs/c_stage_results.md`](../docs/c_stage_results.md) for the complete
tables, communication interpretation, topology caveat and backend decision
guide.

## D1: minimal single-GPU Top-1 MoE

The initial D-stage implementation now routes every token to one local expert,
restores token order, preserves the public GPT interface, and remains disabled
by default for Dense checkpoints. Seven focused correctness tests, the full
26-test regression suite, a BF16 CUDA training smoke, MoE checkpoint recovery,
and legacy Dense checkpoint loading passed. This is correctness evidence only;
capacity, token dropping, balance loss, routing telemetry, and formal
Dense/MoE performance results remain later D-stage work. See
[`d1_moe/README.md`](d1_moe/README.md).

## D2: capacity, token drop and balance loss

Four 20-update TinyStories debug runs changed one routing mechanism at a time
while preserving initialization and all 80 sampled windows. Capacity observation
at factor 1.25 reported 30.75% aggregate overflow without changing routing or
loss. Enforcing the limit dropped 31.81% of tokens across the short run. Adding
a balance-loss weight of 0.01 reduced aggregate load CV from 0.380 to 0.099 and
aggregate drop rate from 31.81% to 19.47%. On the final update, max/mean load
fell from 2.938 to 1.316 and drop rate from 42.19% to 1.66%.

These runs validate mechanism behavior, not convergence or throughput. Raw
per-update metrics, the validated summary, commands and limitations are in
[`d2_moe/README.md`](d2_moe/README.md).

## D3: formal Dense/MoE comparison and stability

Dense and the selected four-expert Top-1 MoE each completed three independent
single-GPU no-Profiler runs on identical TinyStories full windows. Dense reached
47,190 tokens/s at 347.190 ms/update; MoE reached 41,898 tokens/s at 391.047
ms/update. MoE therefore delivered 88.78% of Dense throughput while adding
6,293,504 total parameters but only 2,048 active parameters per token. Median
peak allocated memory increased by 125.8 MiB.

Both variants then completed deterministic 300-update stability runs over the
same 9,600 windows. Final validation LM loss was 3.6436 for Dense and 3.6250
for MoE. MoE aggregate load CV was 0.0361, aggregate drop rate was 1.88%, and
the final 100 updates had no dropped tokens. This one-run quality observation
is not evidence that MoE converges better. Full results and limitations are in
[`d3_moe/README.md`](d3_moe/README.md).

## D4: `torch.compile` compatibility and routing boundaries

Using the same D3 workload and raw eager baselines, three compiled runs per
variant show a sharp split. Dense improves from 347.190 to 240.432 ms/update
(1.444x, +44.40% throughput), while Top-1 MoE changes from 391.047 to 395.580
ms/update (0.989x, -1.15% throughput). All losses remain finite and the formal
window/source invariants pass.

The diagnostic trace and compiler logs explain the negative MoE result:
`bincount` and data-dependent `nonzero` break the graph inside the expert loop,
and changing routed-token shapes trigger recompilation. Compilation removes
many copy calls but leaves CUDA compute essentially unchanged. Full commands,
raw benchmarks, reviewable Profiler summaries and limitations are in
[`d4_moe_compile/README.md`](d4_moe_compile/README.md).

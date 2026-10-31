# RouteScale experiment records

This index follows the current [implementation outline](../docs/RouteScale_实施教程大纲.md):
G = native inference (complete), H = nano-vLLM (partially complete),
I = Mini-SGLang, J = training/inference review. Protocols, reports, file paths and
JSON `stage` fields use the current numbering. The
[migration record](../docs/stage_numbering_migration.md) preserves the old/new
paths and file hashes; measurements and experiment outcomes are unchanged.

H still needs timing-stability replication and H2 multi-GPU TP=2/4 experiments.
H2/I2 multi-GPU work is deferred to a later consolidated rental session.
The original H3 correctness gate retains `performance_execution_allowed=false`;
the separate v5 controlled scope has its own passed performance gate.

## I1: Mini-SGLang single-GPU service baseline

The pinned Mini-SGLang / Qwen3-0.6B HTTP/SSE baseline ran 15 requests with
463 finite-logit batch checks. Cancellation reaches the scheduler and active
table/KV references are released. I1 acceptance remains incomplete: six gaps
cover frontend/detokenizer cleanup, finish reasons, context validation, worker
health and shutdown. The [I1 report](i_minisglang/i1_lifecycle/README.md) retains
all failed preparation attempts, raw request events and the independent audit.
Eleven CPU client tests pass; these are functional results, not online performance.

## H3: same-model controlled performance and timing variability

The new scope compares native KV, paged eager and paged Decode Graph using the
same trained Dense artifact, serving environment and BF16 weights. All 2,592
target logit vectors pass the unchanged error thresholds against FP32 math with
the same BF16-rounded parameter values; 864 eager/Graph vectors are exact.
All paths explicitly disable BF16 reduced-precision GEMM reduction. Six pilots,
36 formal GPU processes and an independent audit passed; v1–v4 failures remain.

Observed replay-throughput median ratios for Graph/eager are 2.92–3.74x, while
TTFT does not improve. Timing variability is material: one native case spans
2.46x across independent processes, and a three-group paired follow-up preserves
the speed direction but does not identify a unique historical cause. These are
controlled fixed-token measurements, not stable production speedups or online SLOs.
See the [complete report, ranges and plot](h_nanovllm/h3_performance_v5/README.md).

## H3: earlier RouteScale Dense integration correctness

The project's trained Dense checkpoint now runs through an explicit nano-vLLM
bridge, retaining learned absolute positions, LayerNorm/GELU and tied weights.
Three independent variants pass 792 full-vocabulary native-BF16 comparisons;
264 eager/Graph vectors are bitwise exact. A fresh-process audit independently
recomputes the native references and verifies 52 state-dict tensor keys,
invalid-request guards, the 512-cached-token boundary and engine reclamation.
This is first-scope correctness, not a performance ranking or full H3 completion.
Both numerical sides use full BF16 weights in the serving environment, so G3's
FP32-weight/autocast timings are not direct controls. See
[`h_nanovllm/h3_routescale/README.md`](h_nanovllm/h3_routescale/README.md).

## H1: pinned nano-vLLM reproduction

The upstream execution path is isolated from the training environment and uses
a pinned Qwen3-0.6B snapshot. H1 passed its full acceptance sequence: 11 upstream
state tests, FP16/BF16 attention checks and two new-process generation requests
with no remaining request/block references. See
[`h_nanovllm/h1_run.json`](h_nanovllm/h1_run.json).
H2 has 24 fixed workloads and 48 passed CPU metadata probes. The original pilot
was stopped before formal timing by a strict BF16 prefix-logit gate; its failed
gate and controlled FP32-attention diagnostic are retained. Batch-1 Graph decode
also failed its numerical gate. Only the three passed batch-8 cases enter the
separate Graph-only scope: 18/18 formal processes passed, with 6.21–8.51x output
throughput in this local offline benchmark and about 8.14 MiB extra peak allocated
memory. See [`h_nanovllm/h2_graph_b8/README.md`](h_nanovllm/h2_graph_b8/README.md).
This does not qualify the excluded batch-1 or prefix cases. Commands, scope and the source
walkthrough are indexed in [`h_nanovllm/README.md`](h_nanovllm/README.md).
The subsequent opt-in corrected policy fixes both numerical comparisons by
unifying page-table width and paged Prefill. All 2,048 full-vocabulary tensor
comparisons across 64 cold/warm steps and eight cases are exact. See
[`h_nanovllm/h2_numerics/README.md`](h_nanovllm/h2_numerics/README.md).
This is correctness evidence, not a new performance ranking; historical gates and timings remain unchanged.
The separately qualified corrected-policy performance pilot passed all 36 formal
processes and its audit: 2,250 requests and 144,000 output tokens. Graph throughput
is 5.90–7.47x eager; a 50% prefix-token hit rate reduces warm TTFT from 137.64 to
71.57 ms, without a corresponding whole-generation throughput gain (0.994x).
These are separate measurements, not replacements for the native results above.
The Graph/Prefix scope alone does not cover subsequent batching,
fragmentation/capacity, TP or model-integration work. Full results and the measurement contract are in
[`h_nanovllm/h2_corrected_pilot/README.md`](h_nanovllm/h2_corrected_pilot/README.md).
The subsequent fixed-cohort versus continuous-admission pilot completed CPU
contracts and GPU preflight, but the mixed-length cases failed the unchanged
numerical budget (max relative RMSE 3.974% versus 0.5%). Formal processes remain
0/18. Cold controls isolate the first divergence to BF16 Prefill output-projection
row shape, not an established KV-address bug; an isolated tile512 linear control
is exact but is not a full-model fix. See
[`h_nanovllm/h2_scheduling/README.md`](h_nanovllm/h2_scheduling/README.md).
The subsequent full-model tile512 strategy passed all 1,664 per-request logit
vectors exactly, then all 18 formal processes and independent verification:
864 measured requests and 29,952 output tokens. Mixed workloads gain about 32%
output throughput and reduce TTFT p95 by 42% / 56%, while ITL p99 rises from about
8.5 ms to 71–75 ms. The equal-length control has no refill opportunity or mechanism
benefit. Both sides use the same opt-in BF16 linear tiling; this is not native
upstream timing or online SLO evidence. Original failure data and its exact source
snapshot remain intact. See
[`h_nanovllm/h2_scheduling_tile512/README.md`](h_nanovllm/h2_scheduling_tile512/README.md).
The subsequent KV allocation study passes 18 CPU capacity points, 13 real-GPU
tensor/address checks and 1,056 exact full-vocabulary vectors across a Decode
page boundary. With the same 896 MiB pool, 128-token live KV fits 32 paged
allocations versus eight full-window reservations; 257-token requests expose
substantial page-tail waste. A separate hole-reuse control distinguishes
contiguous external fragmentation from page tails and unused reservations.
These are allocation/functional results, not throughput or measured maximum
serving concurrency. See
[`h_nanovllm/h2_kv_capacity/README.md`](h_nanovllm/h2_kv_capacity/README.md).
The separate real-model KV-budget pilot passed 4,160 exact numerical vectors
and all 18 formal processes plus saved-logit/raw-metric audit: 864 measured
requests and 74,880 output tokens. In a controlled 448 MiB pool, reducing
reservation allows real cohorts of eight instead of four, delivering 1.950x
short-request and 1.887x page-growth throughput. TPOT p50 is slightly worse.
The 896 MiB ample-budget control admits eight on both sides and records a
negative result: paged throughput is 0.931x, not a universal speedup. Both sides
still use paged attention; this is not a native contiguous-kernel comparison
or a search for full-device serving capacity. See
[`h_nanovllm/h2_kv_pressure/README.md`](h_nanovllm/h2_kv_pressure/README.md).
A separate three-process negative-control diagnosis completed 48 balanced
full-path traces, 27 controlled-page CUDA-stream probes and 3,072 exact saved
full-vocabulary comparisons. The median of process-level paired throughput
ratios is 1.017x; the original 0.931x slowdown was not stably reproduced.
Dense-page Decode stream medians differ by about 1%, while some Prefill
differences remain. This neither replaces the original result nor establishes
a unique historical root cause or a runtime fix. See
[`h_nanovllm/h2_kv_diagnostic/README.md`](h_nanovllm/h2_kv_diagnostic/README.md).

## G1--G3: inference artifact and native KV cache

The first inference baseline exports the trained 50.91M Dense TinyStories
checkpoint without optimizer state, verifies strict loading and FP32/BF16
logits, then compares contiguous-KV Prefill/Decode with full recomputation.
All G1 and G2 correctness workloads pass. G3 completed its full 48-variant,
144-process matrix on one RTX 5060 Laptop GPU. KV caching ranges from slower at
batch 1 to 2.844x output throughput at batch 32 in the tested normal-context
workloads; it loses its benefit when the learned-position window forces a
cache rebuild for every token. Results and reproduction commands are indexed
in [`g_serving/README.md`](g_serving/README.md).

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
- Checkpoint: `runs/checkpoints/legacy/out-tinystories/ckpt.pt`, 585.6 MiB,
  `iter_num=300`.
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
large Chrome traces remain under ignored `runs/traces/b2_profiler/` (109 MiB eager and
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

## E: fixed-capacity dispatch and Triton Grouped GEMM

E-stage replaced the ragged `bincount/nonzero` expert loop with a
deterministic fixed-capacity buffer, then compared `torch.bmm` and a
custom Triton Grouped GEMM. The isolated Triton kernels were 1.50x and 1.32x
faster than the Python expert GEMM loop for the two formal projection shapes.

Across three complete-update runs per variant, eager padded Triton improved
throughput by 8.33% over eager loop. Under `torch.compile`, padded
Triton reached 65,945 tokens/s at 248.451 ms/update, 60.63% higher throughput
than compiled loop and 96.92% of compiled Dense. Compiled padded
`torch.bmm` reached 65,917 tokens/s, so the custom kernel's full-model
advantage was only 0.042%.

Profiler evidence shows that fixed-capacity dispatch, rather than the GEMM
kernel alone, is the decisive change: launch calls fell 28.5%, copy calls
77.4%, and `nonzero`, `bincount`,
`index_select` and `index_copy` disappeared. Compiler
diagnostics emitted no graph-break or recompile record for the padded path.
See [`e_triton_moe/README.md`](e_triton_moe/README.md) for the
protocol, all eight formal variants, limitations and raw evidence map.

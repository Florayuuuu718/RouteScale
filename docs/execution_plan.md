# Remaining execution plan

Agreed on 2026-10-31. This file records work to do, not completed results.
Existing reports, failed attempts, numerical gates and source snapshots remain
the evidence for completed experiments.

Progress on 2026-10-31: **I1 accepted** for TP1, shared tokenizer and
Graph/Overlap off, with seven replayable patches, bounded admission, 40 local
tests and four independently audited real-GPU scenarios (130 checks).
See [the acceptance record](../results/i_minisglang/i1_repaired/acceptance.json)
and [runbook](i1_lifecycle_repairs.md). The bounded I2 FI/TP1 shape-controlled
scope below is also accepted. I3, H3 replication and delivery preparation remain
pending. No paid resources have been used.

I2 v1 has now started. Two replayable I2 patches, 12 CPU tests and seven real
HTTP lifecycle profiles passed (84 requests, 280 checks, including overlap
cancellation). The first full-model numerical gate failed: all 1024 Naive
vectors are finite, but none meet the frozen tolerance against the FP32
reference. Independent diagnostics and the failure are preserved in the
[I2 report](../results/i_minisglang/i2_single_gpu_v1/README.md). Formal performance
processes: **0/21** for that original HF-reference scope. Those failures remain unchanged.

Subsequent localization found two successive first divergences: FA2 automatic
split-KV planning and BF16 GEMM row geometry. A new explicit common policy
(prefill no-split + linear tile512) qualified all 12544 full-vocabulary vectors
exactly, including held-out seed/length/batch inputs; seven real HTTP profiles
passed 280 checks. The separately frozen FI/TP1 performance scope completed
21/21 processes and 160 audit checks. See the [scoped report](../results/i_minisglang/i2_shape_performance_v1/README.md).
Some process CVs reach 21.4%; small gains remain uncertain. The original HF FP32
equivalence is still failed, and these results include the padding cost. FA4 and
TRTLLM were explicitly probed and unavailable in the pinned stack; no fallback
was ranked. TP2/4 remain deferred.

The user then prioritized external HF alignment before I3. The full original HF
FP32 and native BF16 outputs reproduce exactly. The first external divergence is
the layer0 RMSNorm rounding boundary; HF BF16 itself also fails all 1024 vectors
against the original FP32 threshold. See the [precision follow-up](../results/i_minisglang/i2_external_alignment_v1/README.md).
An explicit FP16 main path with FP32 residual and layer/final norm passed all
2048 Naive/Radix vectors at the original threshold (worst relative RMSE 0.322%).
Keep original BF16 failures and scoped performance acceptance separate. Before I3,
freeze the selected precision policy and qualify its scheduling variants and HTTP
lifecycle; a numerical precision candidate alone does not authorize performance.

The seven-profile precision qualification is now independently accepted: mixed
precision passes all 12544 vectors against the original HF FP32 thresholds
(worst relative RMSE 0.3614%); the simultaneous BF16 control has 12544 exact
internal vectors. All 14 HTTP profiles passed 168 requests and 560 functional
checks, but the process-journal gate rejects an unrelated Windows Python GPU
occupancy (92 MiB). Preserve that failed attempt; wait for exclusive GPU, rerun
HTTP in a fresh directory, then complete the 42-process precision/cost matrix
and final audit before I3. The 20 CPU checks passed. See the
[current I2 report](../results/i_minisglang/i2_precision_v1/README.md) and
[retry commands](i2_precision_retries.md). I2 local completion remains pending.

## Local work before renting GPUs

1. **I1 lifecycle acceptance (accepted at the scope above).** Preserve the pinned upstream baseline in
   `results/i_minisglang/i1_lifecycle/attempt004`. Repair frontend request cleanup,
   detokenizer cancellation cleanup, finish reasons, over-context error handling,
   worker health/failure propagation and bounded shutdown. Add bounded admission.
   Deliver replayable patches, focused tests, real single-GPU normal/fault runs,
   request-state and process diagrams, a runbook and an independent audit. A
   client terminal alone does not prove backend termination or resource release.
2. **I2 single-GPU ablations.** Compare Naive/Radix caches with zero/shared prefixes
   and cold/warm caches; then Chunked Prefill, Overlap Scheduling and CUDA Graph
   buckets. Change one variable at a time. Recheck correctness and cancellation
   when scheduling changes. Compare attention backends only when supported.
3. **I3 online loads.** Freeze a common arrival schedule and low/near-saturation/
   overload rates after pilot runs. A first bounded matrix is one mixed workload,
   three rates, baseline versus one policy, three independent service processes:
   18 runs. Freeze latency objectives before comparing goodput. Account for every
   offered request, including failures, timeouts, cancellations and rejections.
   Report SSE chunk intervals as such unless actual token timestamps exist.
4. **H3 timing replication.** Keep the v5 numerical policy and workload; prioritize
   the noisiest case, balance execution order and record temperature, clock and
   power state. Preserve historical observations. Report uncertainty when a
   precise steady-state speedup is not supported.
5. **Local delivery preparation.** Prepare one full serving framework, dependency
   or container instructions, health/readiness, shutdown, monitoring, CPU CI and
   GPU smoke where locally feasible. Check a clean checkout. Prepare all TP
   configurations, correctness checks, timeouts and collection scripts before
   paying for a multi-GPU session. Do not run interfering benchmarks concurrently.

## Consolidated multi-GPU session

After local prerequisites pass, freeze the pending H2/I2 TP=1/2/4 matrix, resource
budget and stopping rules. Re-run TP=1 on the same rented host. Record topology,
NCCL, per-rank memory, numerical checks and computation/communication traces.
No paid resources are authorized by this plan alone. Unrun cases remain pending.

## Final acceptance and learning material

Finish full-framework validation and the J/M7 training-to-serving review. Compare
RouteScale Dense native/recompute/paged paths separately from the common Qwen
nano-vLLM/Mini-SGLang paths. Keep hardware, precision and timing boundaries clear.
Include gains, negative results, limitations, reproduction and fault demonstrations.

After experiments are complete, write the remaining stage reviews/interview guides
in the existing local-only learning area. For each stage record the problem,
mechanism, implementation, evidence, tradeoffs, knowledge to learn and independent
explanation/coding/debugging exercises. Engineering completion does not imply
personal mastery.

## What belongs in Git

Commit source, tests, configurations, execution protocols, source walkthroughs,
reproduction instructions and compact result/audit evidence. Keep model weights,
datasets, virtual environments, caches and large mutable logs under ignored runtime
paths. Personal job-search planning, learning notes and interview practice remain
local. Historical numerical failures and their protocols remain available; they
explain why later scopes and policies differ.

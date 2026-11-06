# Remaining execution plan

Agreed on 2026-10-31. This file records work to do, not completed results.
Existing reports, failed attempts, numerical gates and source snapshots remain
the evidence for completed experiments.

Progress on 2026-10-31: **I1 accepted** for TP1, shared tokenizer and
Graph/Overlap off, with seven replayable patches, bounded admission, 40 local
tests and four independently audited real-GPU scenarios (130 checks).
See [the acceptance record](../results/i_minisglang/i1_repaired/acceptance.json)
and [runbook](i1_lifecycle_repairs.md). **I2 local FI/TP1 is now complete**:
the shape-controlled scope and the subsequent HF-qualified mixed-precision
scope are accepted. **I3 local FI/TP1 is also accepted** at the finite-trace
scope recorded below. The bounded H3 replication investigation is complete,
with residual timing variability explicitly retained. Local delivery and TP preparation
are accepted at the bounded scope below.
No paid resources have been used.

The original I2 v1 attempt is a preserved rejected BF16/HF scope. Two replayable
I2 patches, 12 CPU tests and seven real HTTP lifecycle profiles passed
(84 requests, 280 checks, including overlap
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
The original BF16 failures and scoped performance acceptance remain separate.
That candidate required a frozen policy, seven scheduling variants and HTTP
qualification before performance; the full acceptance is recorded below.

On 2026-11-01 the full seven-profile precision scope is independently accepted: mixed
precision passes all 12544 vectors against the original HF FP32 thresholds
(worst relative RMSE 0.3614%); the simultaneous BF16 control has 12544 exact
internal vectors. After the unrelated Windows workload released the GPU,
http004 passed 168 requests, 560 checks and all 14 process/cleanup gates.
The balanced 42-process performance/cost matrix, 347 independent audit checks,
20 CPU tests and final acceptance passed; 112 statistics cells, 56 precision
costs and 96 system comparisons are exported. All experiment processes exited
and the GPU returned to 0 MiB / 0%. The rejected http001–003 attempts and the
original BF16 HF failure remain preserved. See the
[current I2 report](../results/i_minisglang/i2_precision_v1/README.md) and
[retry commands](i2_precision_retries.md). TP2/4 remains deferred.

I3 completed on 2026-11-01. Its new text workload exposed three numerical
failures under the original I2 precision, so that failed scope and an exact
975-batch/1664-vector replay are preserved. A separately declared precision
policy (FP16 lower 12 layers, FP32 upper 16 layers/head, FP16 FI attention/KV)
passed all 3840 qualification vectors at the unchanged threshold, including
eight new held-out requests per policy. Fresh pilot and lifecycle audits passed.
The balanced 18-process open-arrival matrix accounts for all 864 offers:
744 completions and 120 rejections, with no unexpected failures or timeouts.
All 3889 final audit checks, 26 CPU tests, full exports and final acceptance
passed. See the [I3 report](../results/i_minisglang/i3_online_v2/README.md) and
[precision/online protocol](i3_online_protocol_v2.md). The GPU is idle and no
paid resources were used. This is a finite local workload, not a steady-state
or production-tail claim. The subsequent H3 replication is recorded below.

H3 timing replication completed on 2026-11-02 at the bounded investigation scope.
The v5 source, numerical policy and tokens remain unchanged. A read-only full v5
audit reproduced, and fresh qualification passed 1728 vectors plus 576 exact
eager/Graph vectors. The two noisiest cases completed 36 independent processes,
six balanced path permutations, 20 warmup and 20 measured waves per scope.
All 301 audit checks, 23 CPU tests, exports and final engineering acceptance
passed. All twelve six-process throughput max/min ratios are <=1.247 and all
paired-ratio CVs are <=15%, but 17/72 process scopes exceed the 15% within-process
CV limit; only 3/12 cells meet the full stability rule. Investigation completion
does not establish precise steady-state gains. See the
[replication report](../results/h_nanovllm/h3_timing_replication_v1/README.md).
The GPU is idle, historical evidence is unchanged, and no paid resources were
used. Local framework delivery and TP preparation are now accepted below.

Local delivery completed on 2026-11-03. Mini-SGLang has 131 audited checks;
full SGLang0.5.20 has three native GPU scenarios (75 canonical requests, 705
checks), a separately declared container queue-handshake scope (71 requests,
616 checks), explicit 28-request shape preparation and a real default-entrypoint
Docker stop with init, successful inflight completion and GPU release. The
CPU-only suite passes 54 tests. Native BF16 does not inherit I3 HF qualification.
The offline image, fixed dependencies, source/weight hashes, dashboards, failed
attempts and limits are in [the delivery report](../results/delivery/local_v1/README.md).
TP preparation passed 640 local vectors and 70 checks; the same-host TP1/2/4
matrix has 30 qualifications and 90 three-run performance processes prepared.
Actual multi-GPU qualifications, performance and traces remain unrun. No paid
resources are authorized, and personal reviews/interview guides remain deferred.

## Local work before renting GPUs

1. **I1 lifecycle acceptance (accepted at the scope above).** Preserve the pinned upstream baseline in
   `results/i_minisglang/i1_lifecycle/attempt004`. Repair frontend request cleanup,
   detokenizer cancellation cleanup, finish reasons, over-context error handling,
   worker health/failure propagation and bounded shutdown. Add bounded admission.
   Deliver replayable patches, focused tests, real single-GPU normal/fault runs,
   request-state and process diagrams, a runbook and an independent audit. A
   client terminal alone does not prove backend termination or resource release.
2. **I2 single-GPU ablations (accepted at the local FI/TP1 scope).** Compare
   Naive/Radix caches with zero/shared prefixes and cold/warm caches; then
   Chunked Prefill, Overlap Scheduling and CUDA Graph
   buckets. Change one variable at a time. Recheck correctness and cancellation
   when scheduling changes. Compare attention backends only when supported.
3. **I3 online loads (accepted at the bounded local scope above).** Freeze a common arrival schedule and low/near-saturation/
   overload rates after pilot runs. A first bounded matrix is one mixed workload,
   three rates, baseline versus one policy, three independent service processes:
   18 runs. Freeze latency objectives before comparing goodput. Account for every
   offered request, including failures, timeouts, cancellations and rejections.
   Report SSE chunk intervals as such unless actual token timestamps exist.
4. **H3 timing replication (bounded investigation accepted; residual variability retained).** Keep the v5 numerical policy and workload; prioritize
   the noisiest case, balance execution order and record temperature, clock and
   power state. Preserve historical observations. Report uncertainty when a
   precise steady-state speedup is not supported.
5. **Local delivery preparation (accepted at the local scope above).** Prepare one full serving framework, dependency
   or container instructions, health/readiness, shutdown, monitoring, CPU CI and
   GPU smoke where locally feasible. Check a clean checkout. Prepare all TP
   configurations, correctness checks, timeouts and collection scripts before
   paying for a multi-GPU session. Do not run interfering benchmarks concurrently.

## Consolidated multi-GPU session

F local preparation was also accepted on 2026-11-03 after the user's request:
12 jobs, 18 real worker processes, Gloo world1/2/4, NCCL1 FP32/BF16, D/E expert
shapes and small complete GPT updates passed 62,892 numerical/state checks.
60 CPU contract tests and 17 original MoE regressions passed; 5,204 historical
evidence files stayed byte-identical. The protocol, launcher, independent tensor
audit, eight-job same-host CUDA qualification matrix, data hash preflight and
full-training rental requirements are ready. See the
[F preparation report](../results/f_ep/local_v1/README.md) and
[runbook](f_ep_runbook.md). This completes bounded local preparation, not F's
two-card acceptance: EP2 GPU qualification/performance/communication traces are
all zero. Subsequent full-training preparation passed four additional jobs and
4,996 checks, including Gloo2 accumulated/shared gradients, active global norm
clipping and a real full D/E TinyStories 16,384-token update against the original
GPT. The full training runner and nine-process gated benchmark plus trace entry
point are prepared; the final CPU suite passes 64 tests. See the
[training protocol](f_training_preparation.md). Rerun these qualifiers on CUDA1/2
of the target host before timing; no paid resources are authorized.

After local prerequisites pass, freeze the pending H2/I2 TP=1/2/4 matrix, resource
budget and stopping rules. Re-run TP=1 on the same rented host. Record topology,
NCCL, per-rank memory, numerical checks and computation/communication traces.
Include F EP2 qualification and the separately gated MoE training matrix in the
same authorized session where resources fit; do not conflate EP with serving TP.
No paid resources are authorized by this plan alone. Unrun cases remain pending.

Deployment and session orchestration are documented in the [consolidated rental runbook](rental_execution.md).
The prepared matrix is H2 nano-vLLM (9 qualifications / 27 timing runs), I2
Mini-SGLang (21 / 63), plus F EP qualification and nine complete-training timing
runs. All comparisons rerun the single-GPU reference on the same target host.
C1–C7 multi-GPU acceptance is already complete and is not repeated. The current
TP scope uses Qwen3-0.6B only; larger-model I2 scaling, native-DDP-reducer MoE
timing, full-GPT EP trace and multi-GPU HTTP/I3 are separate extensions.

On 2026-11-06, the remaining local deployment/execution/reporting preparation
was accepted: 4,967 portable bundle files restored and verified, three pinned
environments rebuilt, 85 CPU tests passed, and a real restored Gloo2 tiny GPT
update passed 1,744 checks. Prior numerical sources and 5,361 evidence files
remain unchanged. Failed offline installation and the stopped download are
preserved. See [the acceptance report](../results/rental/local_v1/README.md).
The next dependency is the actual four-GPU host and an explicit cost/time budget,
then target-host qualification; no actual multigpu result is claimed here.

## Final acceptance and learning material

Full-framework local validation is accepted above. Finish the remaining
multi-GPU experiments and the J/M7 training-to-serving review. Compare
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

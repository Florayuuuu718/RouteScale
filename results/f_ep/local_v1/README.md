# F-stage local EP preparation — accepted, two-GPU F acceptance pending

On 2026-11-03 the local preparation passed **12 jobs / 18 worker processes**,
**62,892 tensor/state/ledger checks**, **204 rank-updates** including separate
trace jobs, and **5,507 aggregate checks**. All workers exited successfully and
the GPU returned to 0 MiB / 0%. No paid resources were used.
The [acceptance](acceptance.json) deliberately records `f_stage_complete=false`
and `real_multigpu_accepted=false`.

The additional complete-training preparation also passed four jobs / five
workers and **4,996 checks**: accumulated gradients and active global clipping
in Gloo2/FP32 and NCCL1/FP32/BF16, plus a real **16,384-token full TinyStories
update** in all three implementations. The final CPU contract suite has **64
tests**. Its separate [training acceptance](training/acceptance.json) preserves
the original module scope and still prohibits multi-GPU performance claims.

This work is useful before renting: it resolves routing, gradient scaling,
capacity and optimizer semantics on existing hardware, and provides runnable
two-GPU qualification rather than spending rental time implementing those paths.
The user requested this optional stage's preparation after local M5 completion.

## Completed experiments

| Job | Processes | Rank-updates | Checks | Largest absolute tensor difference |
|---|---:|---:|---:|---:|
| Gloo world1, FP64 | 1 | 21 | 7,843 | 1.54e-10 |
| Gloo world2, FP64 | 2 | 42 | 12,330 | 1.54e-10 |
| Gloo world4, FP64 | 4 | 84 | 21,430 | 1.54e-10 |
| Gloo world2, complete small GPT FP32 | 2 | 2 | 790 | 4.77e-7 |
| NCCL1 FP32 small cases | 1 | 21 | 7,843 | 3.67e-9 |
| NCCL1 FP32 D/E expert shape | 1 | 1 | 248 | 0 |
| NCCL1 FP32 complete small GPT | 1 | 1 | 490 | 0 |
| NCCL1 BF16 small cases | 1 | 21 | 7,843 | 0 |
| NCCL1 BF16 D/E expert shape | 1 | 1 | 248 | 0 |
| NCCL1 BF16 complete small GPT | 1 | 1 | 490 | 0 |
| Separate Gloo2 trace | 2 | 6 | 2,046 | 1.54e-10 |
| Separate NCCL1 CUDA trace | 1 | 3 | 1,291 | 2.33e-10 |

The seven small cases cover balanced and uneven origins, all-to-one routing,
an empty origin, capacity observation, stable global dropping and changing
routes across three AdamW updates. D/E expert shape means 4096 tokens, width512,
four experts, capacity1280; it is a module update, not the complete TinyStories
training benchmark. GPT smoke uses two layers, width16, batch4, sequence8 and
one update; the CPU two-rank experiment checks the complete shared-gradient path.

Inputs, outputs, input gradients, router/expert parameter gradients, updated
weights and AdamW moments/steps are compared with the existing global single-
device ragged-loop MoE. A separate full-expert-replication control matches the
same global admission mask and uses manual gradient SUMs. It is a DDP-semantics
control, not a native PyTorch DDP reducer benchmark. Globally idle experts keep
`grad=None`; previously initialized optimizer state and weights stay unchanged
while those experts are unused. These results do not alter E's documented
padded-empty-expert limitation.

The frozen elementwise tolerances are FP64 2e-6/2e-5, FP32 3e-5/3e-4 and BF16
0.002/0.03 (atol/rtol), including optimizer tensors. BF16 is internal training-
implementation equivalence here; it carries no HF inference qualification.
No tolerance was changed after execution.

## Evidence and preparation

- [Protocol](../../../docs/f_stage_protocol.md),
  [implementation/runbook](../../../docs/f_ep_runbook.md),
  [runtime matrix](../../../config/f_ep_matrix.json) and
  [pending rental requirements](../../../config/f_ep_rental_requirements.json).
- [Acceptance](acceptance.json) binds compact job/rank/audit records and frozen
  [source snapshots](source_snapshot/expert_parallel.py). Every numerical audit
  reopens the raw tensor ledgers and checks hashes, parameter ownership,
  send/receive transpose, order and token conservation. Approximately 912 MiB of
  raw tensors/traces remain under ignored `runs/f_ep/local_v1`, without weights
  or datasets in Git. Compact audit files retain their SHA256 values.
- [Trace inventory](trace_inventory.json) binds three full traces: Gloo2 ranks0/1
  and CUDA NCCL1 rank0. Rank reports retain `ep/` scopes and collective counts.
  These contain profiled/cold-start overhead; CPU Gloo has no GPU-link timing,
  NCCL1 has no inter-GPU traffic. Audit payload tables report logical activation
  and expert-ID bytes, excluding count/router collectives, not measured wire bytes.
- [Negative controls](negative_controls.json) rejected wrong output, wrong split
  counts and replacing an absent expert gradient with zero. Their SHA256 metadata
  was intentionally rebound, so numerical/semantic checks caused rejection.
  [Control source](check_audit_controls.py) is retained for replay in `runs/f_ep`.
- [CPU receipt](cpu_checks.json): 60 delivery/F guard tests and 17 existing MoE
  regressions passed. A dedicated
  [CPU Gloo CI workflow](../../../.github/workflows/f-ep-cpu.yml) installs pinned
  CPU Torch and runs real world1/2/4 and complete small GPT checks. The existing
  small serving CI remains free of Torch.
- [Dataset preflight](dataset_preflight.json) streamed and verified the actual
  TinyStories train/validation hashes against the prepared rental contract.
  This read-only check is not a new TinyStories training result.
- [Historical inventory](historical_before.json): all **5,204 existing evidence
  files** stayed byte-identical; only the existing `results/README.md` index is
  intentionally editable. No old numerical failure or source snapshot changed.

Preparation issues remain in [the issue directory](preparation_issues/): resolving
the venv interpreter symlink launched its base Python without Torch; using system
Python for the serving suite lacked httpx. The launcher now preserves the venv
executable path, and the qualified serving suite uses its pinned CPU environment.
Neither issue was a numerical failure or a reason to change a gate.

## Complete training, accumulation and clipping

The subsequent [training protocol](../../../docs/f_training_preparation.md) and
[runner](../../../scripts/prepare_f_training.py) close the remaining local
implementation prerequisites. They compare the original GPT, the full-expert
manual-reduction replica and EP with identical initialization and token windows.
They SUM shared gradients after accumulation, count replicated parameters once
in global norm clipping, preserve absent expert gradients and compare every
gradient, weight and AdamW state. The independent audit reconstructs the clipped
norm directly from reference gradient tensors; the small cases force clipping
with max_norm0.01.

| Additional job | Worker ranks | Audit checks | Largest absolute difference |
|---|---:|---:|---:|
| Gloo2 FP32 complete small GPT, two accumulations | 2 | 1,744 | 9.54e-7 |
| NCCL1 FP32 small GPT, two accumulations | 1 | 967 | 0 |
| NCCL1 BF16 small GPT, two accumulations | 1 | 967 | 0 |
| NCCL1 BF16 complete D/E TinyStories model | 1 | 1,318 | 5.88e-4 |

The full-model job uses the original 57,461,760-parameter MoE, eight layers,
width512, sequence512, global microbatch8 and four accumulations, capacity1.25,
drop/balance, AdamW and clip1.0. Each of reference/replica/EP performs one actual
update on the same frozen TinyStories windows. Largest absolute difference is
across all tensors/state and passes the predeclared BF16 0.002/0.03 gate; this
does not assert bitwise equivalence. Large reference/state files remain local.

The training launcher prepares **nine gated performance processes** (three
repeats each of one-GPU reference, two-GPU replica and EP2). It requires same-host
module and complete-model CUDA1/2 qualifications, matching source/raw hashes,
and idle distinct GPUs. The separate
[two-GPU trace entry point](../../../scripts/collect_f_ep_trace.py) applies that
same gate before collecting D/E-shape activation/gradient All-to-All traces.
Neither entry point rents resources. Those pending experiments have not run.

## Remaining F acceptance

There were **zero real EP2 GPU qualifications, zero multi-GPU performance runs
and zero inter-GPU traces**. Run all eight same-host CUDA EP1/EP2 module/GPT
FP32/BF16 qualification jobs after separate resource authorization. Local CPU or
EP1 acceptance cannot open the multi-GPU gate.

Then rerun the prepared full-shape/complete TinyStories update, accumulated
gradients and global clipping qualifiers on CUDA1/2 of that host. The prepared
rental contract carries the D/E model, data hashes, equal 16,384 global tokens
per update, three independent processes, 20 warmup/100 measured updates,
per-rank memory, separate traces and stop rules. Full-model two-card throughput,
native DDP reducer timing and communication cost remain pending. No EP speedup,
memory advantage or TinyStories quality improvement is inferred from local
correctness. Personal reviews and interview guides remain deferred.

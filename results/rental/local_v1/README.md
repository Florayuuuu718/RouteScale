# F / H2 / I2 rental preparation acceptance

Accepted locally on 2026-11-06. [Acceptance](acceptance.json) records
`local_preparation_complete=true`, `real_multigpu_accepted=false` and
`paid_resources_used=false`. This closes deployment, sequential execution and
result aggregation preparation; actual multi-GPU experiments remain pending.

The [execution runbook](../../../docs/rental_execution.md) maps each experiment
to its stage and provides packaging, restore, installation, launch and reporting
commands. C1–C7 already have multi-GPU acceptance and are not repeated.

## What was actually verified

- A portable source/vendor/data/model/CUDA-toolkit bundle was built and restored
  into a fresh checkout. All 4,967 files passed the bundle hashes. The package is
  22,812,790,808 bytes (about 21.2 GiB), at `runs/rental/bundle002`; its source
  commit is `7b4018a3339d09122987136d9146e537a408ec2d`. The final reporting commit
  is later than the packaged source; no numerical core was changed.
- Training, Mini-SGLang and nano-vLLM environments were recreated with
  Python3.11.16 and separate locks. All three dependency checks and exact-version
  checks passed. The final installation used package-local H1 wheels and the
  existing uv cache with network disabled. The package itself does **not**
  contain every dependency wheel; a fresh rental host needs network or a
  populated cache. [Environment record](environments.json).
- The relocated CUDA13.0 compiler compiled a CUDA translation unit without
  launching a GPU kernel. Torch, FlashInfer and FlashAttention imported in the
  recreated environments. FlashInfer initializes CUDA during import; this is
  recorded in [imports.json](imports.json), not described as CPU-only import.
- 21 new CPU tests and 64 existing delivery tests passed. The new tests cover
  host/budget rejection, changed bundle files, wrong/missing work, warmup
  exclusion, three-process statistics, trace overlap, changed logs/windows,
  resume guards and stopping after a failed stage. Real subprocess tests verify
  timeout cleanup, detached grandchildren and immediate-parent-exit orphans,
  while leaving unrelated processes alive. The final restored checkout also
  passed the same 21 tests and static checks.
- A real Gloo2 tiny complete-GPT update ran in the final restored training
  environment. Its 1,744 independent gradient/weight/AdamW-state checks passed,
  including accumulated gradients and global clipping. See
  [the tensor audit](gloo2_audit.json). It is deployment validation, not a new
  full-size training throughput experiment.
- 5,361 pre-existing evidence files stayed byte-identical. Previously accepted
  local delivery and F numerical scopes remain intact; no old GPU matrix was
  repeated. Both runner locks are free. After the import checks, NVML reported
  12 MiB / 0% with no compute processes; this receipt does not fabricate a
  zero-memory reading. No new GPU qualification or benchmark jobs were run.

The full result-export path was tested with explicitly synthetic TP and F
reports, including all 90 TP and nine F timing jobs and 72 per-rank trace files.
Only qualification tensor gates are substituted in that unit fixture. These
values are software test data and are not published as experimental results.

## Preserved preparation failures

The first restored checkout could rebuild training and Mini but lacked nano's
Torch/NCCL/Triton artifacts in the uv offline cache. Its logs and hashes remain
under `runs/rental/rehearsal001` and [offline_attempt.json](offline_attempt.json).
A second attempt began downloading those artifacts and was explicitly stopped
after finding the already preserved H1 wheels; it is not counted as successful.
The updated package pins those wheel hashes and uses `--find-links`. The third
fresh checkout completed the full installation and is the accepted deployment.
See [online_attempt.json](online_attempt.json).

An initial import probe incorrectly required every package import to avoid CUDA
initialization. FlashInfer violated that probe assumption, while its import
succeeded. The corrected probe records the observed initialization and actual
subprocess exit. The rejected assumption remains in
[import_probe_initial_failure.json](import_probe_initial_failure.json).

## Remaining real multi-GPU work

| Stage | Qualification | Independent timing runs | Separate traces |
|---|---:|---:|---:|
| H2 nano-vLLM, TP1/2/4 | 9 | 27 | 9 jobs |
| I2 Mini-SGLang, TP1/2/4 | 21 | 63 | 21 jobs |
| F EP | 8 module/GPT jobs + 5 training jobs | 9 | 1 two-rank module job |

The 5 F training jobs include CPU2 tiny, CUDA1 tiny FP32/BF16 and CUDA1/2 full
BF16. Common preflight also checks each Torch stack, topology and NCCL1/2/4.
All timing requires fresh same-host numerical qualification. TP uses the fixed
Qwen3-0.6B workload; larger models and multi-GPU HTTP/I3 are additional scopes.
F's replicated control uses manual gradient reductions. Its trace profiles the
module qualification harness (reference, replicated and EP), so the whole trace
does not measure an isolated EP communication fraction or full-GPT attribution.

The host and budget template remains unfilled. These preparation results do not
start a cloud instance or authorize an unspecified expense. The controller stops
experiments at its deadline; stopping cloud billing still requires releasing the
instance after evidence export. Reviews and interview material remain deferred.

# F EP preparation and reproduction

The [frozen protocol](f_stage_protocol.md) and
[matrix](../config/f_ep_matrix.json) define the local scope. Use the original
training environment (`.venv`, Python3.11, torch2.12.1+cu129 here), rather than
assuming serving-framework environments have identical training arithmetic.
Do not edit historical D/E evidence or reuse its throughput numbers as EP data.

## Ownership and gradient flow

```mermaid
flowchart LR
  X[Local tokens] --> R[Replicated router]
  R --> C[Global counts and stable capacity]
  C --> P[Pack by owner]
  P --> A[All-to-All inputs and expert IDs]
  A --> E[Owned experts only]
  E --> B[All-to-All outputs]
  B --> O[Restore local order and apply gate]
```

The backward executes the inverse output exchange, local expert backward and
inverse input exchange. All origins contribute to each owned expert's gradient.
Only router/shared replicated parameters are SUM-reduced. Local losses use the
global token denominator; an extra world-size averaging would under-scale them.
Zero-token ranks still participate in both backward exchanges. Globally unused
expert parameters are not attached to dummy computations. Optimizer absence
therefore matches the original ragged-loop reference rather than E's padded
empty-expert behavior.

Global capacity is `ceil(factor * global_tokens / num_experts)`. A token's slot
includes counts from earlier ranks; clipping each rank separately is incorrect
for the single-global-batch comparison. Balance loss uses the complete pre-drop
histogram. The separate fully replicated control computes local tokens with
this same global admission mask and manually sums gradients. Its results qualify
DDP semantics, without claiming native DDP reducer performance.

## Local commands

Run from a clean checkout. Weights and TinyStories files are not required for
these deterministic synthetic qualification inputs. Prepare a fresh ignored
runtime path; commands refuse to overwrite results:

```bash
python3 scripts/prepare_f_ep.py --output runs/f_ep/planned.json
python3 scripts/prepare_f_ep.py --execute --output runs/f_ep/replay001
```

`--labels cpu1 cpu2 cpu4` selects CPU jobs. CUDA jobs require idle hardware and
hold the project's global GPU lock. No rank shares a CUDA device. The launcher
never rents resources. It fixes torchrun restarts to zero, bounds execution and
retains failed attempts. Worker logs, tensor ledgers and full Profiler traces
stay under ignored `runs/`; compact reports, audits, provenance and source
snapshots belong in the new F result scope. Real tensors are reopened using:

```bash
.venv/bin/python scripts/audit_f_ep.py runs/f_ep/replay001/cpu2 --output runs/f_ep/recheck_cpu2.json
```

The audit compares every owned parameter, gradient and AdamW state with the
global single-device reference, as well as the full-expert replicated control.
It checks send/receive matrix transposition, output order, global token totals,
expert ownership and stable unused-expert weights/state across updates. Raw
SHA256 values bind the compact audit to the local tensors; tensor files are
not committed. Independent audit failures are retained and stop the matrix.

Trace jobs run separately from qualification. `ep/` scopes distinguish count
gather, split exchange, expert IDs, activation/gradient exchanges and local
compute. CPU Gloo traces explain software sequencing. A CUDA world1 trace
only describes degenerate NCCL1; neither measures inter-GPU communication.
Payload tables count logical activation/ID bytes, exclude count and router
collectives, and are not measured network traffic.

## Two-GPU session after separate resource authorization

On the same target host first record GPU inventory/topology, driver, exact
Torch/CUDA/NCCL versions and idle state. Use the existing
[host preflight](../scripts/preflight_multigpu.py), targeting a new F output path.
Copy source/configuration, preserve its hashes, and use a fresh environment
lock on that host. The target GPUs must be supported by that exact Torch build.

```bash
python3 scripts/prepare_f_ep.py --host-qualification --output runs/f_ep/host_plan.json
python3 scripts/prepare_f_ep.py --host-qualification --execute --output runs/f_ep/host001
```

The eight-job host matrix reruns CUDA world1 and world2 in FP32 and BF16,
including both module adversarial cases and the small complete GPT update.
`qualification_gate` refuses missing EP2, CPU substitution, failed audits,
changed source/configuration and changed raw evidence. Local acceptance cannot
unlock a multi-GPU performance claim.

Then freeze the complete TinyStories GPT benchmark and its identical global
windows, batch/accumulation and initialization for single-GPU MoE, replicated
MoE and EP2. Keep global capacity and loss scaling identical, collect three
independent unprofiled processes per variant, per-rank memory and separate
NCCL computation/communication traces. Native DDP reducer behavior, full GPT
multi-GPU training throughput, multi-GPU failure recovery and communication
overlap remain additional experiments; this local preparation does not claim
their completion. Stop on numerical mismatch, OOM, occupied hardware or timeout;
never weaken a frozen gate to spend the remaining rental time.

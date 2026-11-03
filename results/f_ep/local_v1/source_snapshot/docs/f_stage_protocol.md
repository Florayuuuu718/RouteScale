# F-stage local preparation v1

Frozen before execution on 2026-11-03. D/E are complete. F is optional in the
original outline; the user has now requested its preparation and feasible local
experiments. No paid resources are authorized. Historical D/E/G/H/I records stay
unchanged.

## Scope and gates

Implement an eager Top-1 EP layer with a replicated router and contiguous expert
ownership (`expert // experts_per_rank`). Use real `all_to_all_single` for counts,
expert IDs, inputs, outputs and the inverse exchanges in backward. CPU Gloo
world=1/2/4 exercises real processes and collectives. CUDA NCCL world=1 exercises
the actual GPU path but cannot establish cross-device correctness or speedup.
No top-k, overlap, compile or Triton optimization is included.

Compare against the existing `Top1MoE(dispatch_backend="loop")` using the same
weights, concatenated rank inputs, cotangents and three consecutive AdamW updates
(lr=0.001, betas=0.9/0.95, weight_decay=0.1, zero_grad(set_to_none=True)). Check
outputs, input gradients, router/expert gradients, optimizer moments/steps,
updated weights, counts, drops and order. The loss is a sum of token dot products
divided by the global token count, plus 0.01 times global balance loss.
Router gradients are SUM-reduced; experts already receive gradients from all
origins and must not be reduced a second time. The replicated-full-expert control
SUM-reduces gradients with an explicit globally-active mask, including ranks
where the local gradient is absent. This is a manual DDP-semantics control,
not a measurement of PyTorch DDP reducer performance.

Capacity uses **global** token count. Keep tokens in rank order, then flattened
local order. Rank-local capacities would define a different algorithm. Balance
loss uses global pre-drop hard frequencies and a rank-local probability-sum
contribution; summing rank losses recovers the reference. Globally unused
experts retain grad=None, optimizer state absent (or unchanged after prior use),
and unchanged weights. A zero tensor gradient is a different AdamW contract.

Cases: balanced, uneven origins, all-to-one (empty expert owners), an empty
origin, capacity observation without dropping, capacity dropping, and changing
routes across updates. CPU uses FP64 expert math and the original FP32 router
softmax; tolerances atol=2e-6/rtol=2e-5. CUDA FP32 uses atol=3e-5/rtol=3e-4;
BF16 autocast uses atol=0.002/rtol=0.03. These are elementwise gates for all
numeric tensors including optimizer states, fixed before runs. Shapes/seeds
and case definitions are in the versioned matrix. No passing flag substitutes
for reopening raw tensors and independently checking them.

Local GPU work additionally covers the D/E expert shape (4096 tokens, width512,
four experts, capacity factor1.25) in FP32 and BF16, plus a small complete GPT
forward/backward/update equivalence smoke at EP1. A CPU two-rank GPT smoke
also checks shared-parameter gradient SUMs and global loss normalization.
Synthetic inputs qualify
mechanisms; they are not TinyStories quality or end-to-end throughput evidence.
The formal module shape and complete GPT smoke each perform one optimizer
update; the seven small adversarial cases each perform three consecutive updates.

## Execution, provenance and bounds

Use one GPU per CUDA rank; reject insufficient or occupied GPUs before launch.
Hold `runs/tmp/h2_runner.lock` across GPU work. Process-group timeout=60s,
whole-job timeout=300s (formal shape600s). On timeout terminate the launched
process group and retain its logs. Each output directory is exclusive-create.
Record source/config hashes, exact command, software, per-rank device, topology,
raw tensor hashes, rank reports, exit code and GPU release. CPU profiling is
separate from numerical qualification and cannot be called a GPU/NCCL trace.

Reference capacities are computed over the global batch. The manual replicated
control therefore also applies the same global stable drop mask; ordinary
rank-local D-stage dropping is not an equivalent DDP baseline.

## Consolidated two-GPU window (pending)

On one authorized host, rerun EP1 and EP2 numerical qualification for FP32 and
BF16 with the same configuration/source. Reject timing until every required
rank audit passes. Collect NCCL version, topology and per-rank peaks. Then run
three independent unprofiled processes for single-device reference, full-expert
replication and EP2 at equal global tokens, and separate communication traces.
Compare full GPT updates on identical TinyStories windows before making any
end-to-end claim. Local Gloo, NCCL1 and module shapes cannot complete F's
two-card acceptance. Resource budget and stop rules still require agreement.

The collective API and optimizer absence semantics follow the primary
[PyTorch distributed documentation](https://docs.pytorch.org/docs/2.9/distributed.html#torch.distributed.all_to_all_single)
and [AdamW zero_grad documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.zero_grad).
The installed runtime version and implementation hashes are captured separately.

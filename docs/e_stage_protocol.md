# E-stage Triton Grouped GEMM protocol

## E0: frozen scope

E-stage optimizes the single-GPU Top-1 MoE implementation established in D.
It does not add Expert Parallel or All-to-All. The formal workload remains:

- one RTX 5060 Laptop GPU, BF16;
- TinyStories full;
- 8-layer, 8-head, 512-dimensional GPT;
- batch size 8, sequence length 512, four accumulation steps;
- one four-expert Top-1 MoE block at zero-based layer 4;
- capacity factor 1.25, token drop enabled, balance-loss weight 0.01;
- 20 warmup and 100 measured updates in three independent processes.

The changed variables are dispatch backend and compile mode. Data seed,
initialization, optimizer, training work, measurement boundaries, and loss
definition remain fixed.

## E1: standalone Grouped GEMM

The kernel receives left tensors shaped ``[experts, M, K]`` and
right tensors shaped ``[experts, K, N]``. It schedules all
expert matrices in one Triton launch and accumulates with FP32.

Formal MoE shapes are:

- first projection: ``[4, 1280, 512] × [4, 512, 2048]``;
- second projection: ``[4, 1280, 2048] × [4, 2048, 512]``.

Forward results and input/weight gradients must match `torch.bmm`
within BF16 tolerance. The first training implementation uses Triton forward
and PyTorch batched-GEMM backward; backward kernels are a separate optimization.

## E2: fixed-capacity dispatch

`padded_torch` removes per-expert `nonzero` and the Python
expert loop. Tokens are assigned deterministic positions inside a fixed
`[experts, capacity, hidden]` buffer. Overflow tokens map to a sentinel
row, produce zero MoE output, and retain the residual path.

It must match the D-stage loop backend for:

- expert counts, processed counts, overflow and drop masks;
- output and token order;
- input, router, and expert parameter gradients;
- LM, balance, and total loss;
- state-dict parameter names and counts.

## E3: Triton integration

`padded_triton` changes only the two batched expert matrix
multiplications from `torch.bmm` to Grouped GEMM. Packing, GELU,
gate weighting, unpacking, loss, and optimizer semantics stay the same.

Three nested benchmarks are required:

1. GEMM-only microbenchmark;
2. MoE implementation smoke and profiler diagnosis;
3. complete GPT training update.

## E4: compile and training compatibility

Both padded backends must complete BF16 forward, backward, optimizer update,
checkpoint-compatible state-dict loading, and `torch.compile` smoke.
Compiler logs must confirm that D4's per-expert `nonzero` and Python
loop boundary no longer appears in the padded path.

## E5: formal decision matrix

The formal matrix is:

| Dispatch | Eager | Compiled |
|---|---|---|
| Dense control | three runs | three runs |
| D-style ragged loop | three runs | three runs |
| fixed-capacity `torch.bmm` | three runs | three runs |
| fixed-capacity Triton Grouped GEMM | three runs | three runs |

Formal timing runs without Profiler or per-update routing telemetry. Raw JSON
must record source hashes, data-window hashes, all three run medians, memory,
loss finiteness, environment, and complete configuration.

Profiler and compiler logs are explanatory evidence only. The optimization is
accepted only if the complete training update improves; a faster isolated GEMM
is not sufficient.

The padded paths are frozen to BF16, `dropout=0`, and positive
capacity in E-stage. An expert that receives zero accepted tokens has an
explicit zero gradient in the padded batched computation, whereas the ragged
loop can leave that expert gradient absent; decoupled optimizer weight decay
therefore differs in that edge case. The formal balanced workload routes tokens
to every expert, but exact empty-expert optimizer semantics remain future work.

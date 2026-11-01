# RouteScale

RouteScale is a reproducible Transformer training performance project built on nanoGPT and TinyStories. The project studies how profiling, distributed training, sparse Mixture-of-Experts models and GPU kernels affect end-to-end training performance.

## Verified Baseline

The current baseline uses:

- TinyStories dataset
- 50.91M-parameter Transformer
- 8 Transformer blocks and 8 attention heads
- Embedding width 512 and sequence length 512
- BF16 mixed-precision training
- NVIDIA GeForce RTX 5060 Laptop GPU

A 300-update stability run reduced validation loss from 10.9094 to 3.5545 without NaN or CUDA out-of-memory errors.

With a micro-batch size of 8 and 16,384 tokens per optimiser update:

- Peak allocated GPU memory: 3,899.2 MiB
- Peak reserved GPU memory: 5,148.0 MiB
- Checkpoint saving and training resumption were verified

Detailed experiment records are available in [`results/README.md`](results/README.md).

## Roadmap

Stage labels follow the [implementation outline](docs/RouteScale_实施教程大纲.md):
G = native inference (complete), H = nano-vLLM (partially complete),
I = Mini-SGLang, J = training/inference review. The former standalone G summary
stage was removed. Protocols, scripts, configurations, tests, artifact paths and
JSON stage labels use the current numbering; see the
[migration record](docs/stage_numbering_migration.md).

- [x] Reproducible TinyStories training baseline
- [x] Mixed-precision and micro-batch memory evaluation
- [x] Checkpoint save and resume validation
- [x] Reproducible single-GPU no-Profiler benchmark
- [x] PyTorch Profiler bottleneck analysis and `torch.compile` A/B
- [x] Deterministic DDP window sharding and local C-stage validation
- [x] C5 deterministic full-window coverage and exact checkpoint resume
- [x] DeepSpeed ZeRO-0/1/2/3 single-GPU engine/checkpoint smoke tests
- [x] FSDP2 `fully_shard` and DCP single-GPU checkpoint smoke test
- [x] 1/2/4-GPU DDP strong/weak scaling and communication trace
- [x] Four-GPU DeepSpeed/FSDP2 performance, capacity and checkpoint matrix
- [x] Top-1 Mixture-of-Experts
- [x] Fixed-capacity MoE dispatch and Triton Grouped GEMM evaluation
- [x] Versioned Dense inference artifact with strict-load/logits equivalence
- [x] Native contiguous KV-cache correctness for prefill and incremental decode
- [x] Single-GPU CUDA serving benchmark for TTFT, decode throughput and KV-cache memory
- [x] Pinned nano-vLLM / Qwen3-0.6B reproduction and GPU functional checks
- [x] nano-vLLM CUDA Graph / Prefix Cache pilot A/B (corrected policy: 2,048 exact numerical checks, 36/36 formal processes and audit passed; remaining H-stage work is not implied complete)
- [x] nano-vLLM fixed-cohort / continuous-admission pilot (tile512 policy: 1,664 exact-logit comparisons, 18/18 formal processes and independent audit; original failure retained)
- [x] KV allocation capacity / fragmentation controls (18 CPU capacity points, 13 real-GPU tensor/address checks, 1,056 exact boundary-logit comparisons; not throughput or maximum serving concurrency)
- [x] Controlled KV-budget real-model admission/throughput pilot (4,160 exact vectors, 18/18 formal processes and audit; ample-budget negative result retained, no eviction or online SLO claim)
- [x] Bounded negative-control follow-up (48 balanced full-path traces, 27 CUDA-stream probes, 3,072 exact vectors; historical slowdown not stably reproduced, no runtime fix claimed)
- [x] RouteScale Dense / nano-vLLM first-scope integration correctness (792 native-BF16 logit vectors, 264 exact eager/Graph vectors and independent weight/API/lifetime audit; initial scope excludes timing)
- [x] H3 same-model controlled performance matrix (2,592 BF16 target vectors, 864 exact eager/Graph vectors, 6/6 pilots, 36/36 formal processes and independent audit; explicit GEMM reduction policy, fixed-token replay)
- [ ] H3 timing-stability replication before claiming precise steady-state speedups
- [ ] H2: multi-GPU TP=2/4 ablations (deferred to the consolidated rental session with I2)
- [ ] I: Mini-SGLang online serving experiments
- [x] Mini-SGLang pinned single-GPU HTTP/SSE baseline (15 requests, 463 finite-logit checks; six lifecycle acceptance gaps retained in the [I1 report](results/i_minisglang/i1_lifecycle/README.md))
- [x] I1: lifecycle cleanup, bounded admission, validation, health and shutdown acceptance (TP1, shared tokenizer, Graph/Overlap off; four real-GPU scenarios, 130 audit checks and 40 local tests; [report](results/i_minisglang/i1_repaired/README.md))
- [ ] I3: online load experiments; I2 TP=2/4 deferred to the later rental session
- Original I2 v1 BF16/HF scope (historical rejection): seven HTTP profiles passed (84 requests, 280 checks; 12 CPU tests), but the full-model numerical gate failed. Its formal performance count stays 0/21; see [the preserved failure and diagnostics](results/i_minisglang/i2_single_gpu_v1/README.md). The subsequent mixed-precision scope is accepted below.
- [x] I2 bounded FI/TP1 shape-controlled scope: two successive numerical divergences localized; 12,544 exact vectors, seven HTTP profiles and 21/21 formal processes accepted. Original BF16-to-HF-FP32 equivalence remains failed; padding cost and uncertainty are explicit in [the scoped report](results/i_minisglang/i2_shape_performance_v1/README.md). The [external HF precision follow-up](results/i_minisglang/i2_external_alignment_v1/README.md) motivated the final precision scope below; TP=2/4 remains deferred.
- [x] I2 local FI/TP1 final precision scope: seven mixed profiles pass 12,544 vectors at the original HF FP32 threshold (worst relative RMSE 0.3614%); 12,544 exact BF16 internal-control vectors, 168 HTTP requests / 560 checks, 42/42 balanced performance processes, 347 independent audit checks and 20 CPU tests accepted. Full precision costs and uncertainty are in [the final report](results/i_minisglang/i2_precision_v1/README.md); I3 is next.

The native-weight integration, architecture mapping and remaining measurement
boundaries are documented in
[`results/h_nanovllm/h3_routescale/README.md`](results/h_nanovllm/h3_routescale/README.md).
The subsequent [H3 controlled performance report](results/h_nanovllm/h3_performance_v5/README.md)
includes numerical-policy failures, the complete matrix, independent-process
ranges and a paired timing diagnosis. Graph improves observed replay throughput;
TTFT does not improve, and substantial timing variability limits precise rankings.

The frozen D-stage single-GPU experiment contract and D1/D2 mechanism boundaries
are recorded in [`docs/d_stage_protocol.md`](docs/d_stage_protocol.md). D1
correctness and CUDA smoke evidence is in
[`results/d1_moe/README.md`](results/d1_moe/README.md); capacity, token-drop and
balance-loss evidence is in
[`results/d2_moe/README.md`](results/d2_moe/README.md). The formal Dense/MoE
benchmark and 300-update stability comparison are in
[`results/d3_moe/README.md`](results/d3_moe/README.md). The D4 compile A/B and
dynamic-routing graph-break diagnosis are in
[`results/d4_moe_compile/README.md`](results/d4_moe_compile/README.md).
The E-stage fixed-capacity dispatcher, Triton Grouped GEMM kernel, eight-variant
formal matrix and compiler/profiler evidence are in
[`results/e_triton_moe/README.md`](results/e_triton_moe/README.md).
The best compiled MoE reaches 65,945 tokens/s, 96.92% of the current compiled
Dense control; the large gain comes from stable padded shapes, while custom
Triton and compiled `torch.bmm` are effectively tied end to end.

The completed C-stage findings, decision guide, limitations and evidence map
are documented in [`docs/c_stage_results.md`](docs/c_stage_results.md). The
verified AutoDL evidence snapshot is stored under
[`artifacts/c_stage_autodl/`](artifacts/c_stage_autodl/).

## Repository Structure

- `config/`: reproducible training configurations
- `data/`: dataset preparation code and local tokenized datasets
- `docs/`: protocols, runbooks, reports and the project roadmap
- `notebooks/`: sizing and scaling-law notebooks
- `serving/`: checkpoint export, correctness baselines and native inference code
- `results/`: small, reviewable experiment metrics and reports committed to Git
- `artifacts/`: frozen evidence snapshots imported from completed remote runs
- `runs/`: ignored machine-local checkpoints, profiler traces, logs and archives
- `train_coverage.py`: deterministic C5 coverage/checkpoint entry point
- `train_deepspeed.py`: DeepSpeed ZeRO entry point
- `train_fsdp2.py`: PyTorch FSDP2 entry point
- `distributed_common.py`: shared model/data/optimizer semantics
- `distributed_benchmark.py`: shared C6/C7 no-Profiler timing and rank aggregation
- `moe.py`: single-GPU Top-1 router and local experts
- `triton_grouped_gemm.py`: fixed-shape grouped expert GEMM kernel
- `scripts/preflight_multigpu.py`: target-host admission report and NCCL check
- `scripts/summarize_c4_profiler.py`: per-rank collective trace summary
- `scripts/summarize_c_backends.py`: DeepSpeed/FSDP2 formal benchmark summary

Generated files follow one output contract: lightweight JSON/TXT/Markdown
evidence goes to `results/`; large mutable runtime state goes to `runs/`; a
curated immutable evidence package goes to `artifacts/`.

## Attribution

RouteScale is built on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). The original upstream commit is recorded in `UPSTREAM_COMMIT.txt`.

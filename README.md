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
- `data/tinystories/`: TinyStories preparation scripts
- `results/`: experiment configurations and results
- `docs/`: environment records, project plan and upstream documentation

## Attribution

RouteScale is built on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). The original upstream commit is recorded in `UPSTREAM_COMMIT.txt`.

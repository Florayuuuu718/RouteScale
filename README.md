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
- [ ] PyTorch Profiler bottleneck analysis
- [ ] 1/2/4-GPU DDP scaling
- [ ] Top-1 Mixture-of-Experts
- [ ] Triton kernel evaluation

## Repository Structure

- `config/`: reproducible training configurations
- `data/tinystories/`: TinyStories preparation scripts
- `results/`: experiment configurations and results
- `docs/`: environment records, project plan and upstream documentation

## Attribution

RouteScale is built on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). The original upstream commit is recorded in `UPSTREAM_COMMIT.txt`.


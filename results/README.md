# RouteScale experiment records

## A2: local TinyStories candidate

Date: 2026-09-20. These runs validate the local training configuration on one
NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB). They are not formal throughput
benchmarks because CUDA timing boundaries have not yet been synchronized.

Data: `tinystories_full`, revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

- `train.bin`: 473,992,236 tokens, SHA256
  `66c5c49a38ebcab576854f2f58fad55e06b2bb29ecbc34fa421910e4beb37500`
- `val.bin`: 4,765,918 tokens, SHA256
  `f0d47c000fdf2f7fbae1159832002b079a338cfde9ce4c5f65413da68386f5be`

Candidate model: 8 Transformer Blocks, 8 attention heads, embedding width 512,
sequence length 512, BF16 autocast, `compile=False`, gradient accumulation 4,
50.91M non-position-embedding parameters.

### Micro-batch fit check

Each run performed 20 optimizer updates. Step times are diagnostic only.

| Micro-batch | Tokens/update | Peak allocated | Peak reserved | Typical logged step |
|---:|---:|---:|---:|---:|
| 4 | 8,192 | 2426.3 MiB | 2772.0 MiB | ~184 ms |
| 6 | 12,288 | 3157.7 MiB | 3968.0 MiB | ~264 ms |
| 8 | 16,384 | 3899.2 MiB | 5148.0 MiB | ~346 ms |

Micro-batch 8 was selected. Its peak reserved memory was about 63% of device
memory, leaving headroom for evaluation, profiling, and runtime variation.

### Stability run

Configuration: `config/train_tinystories.py`; exactly 300 optimizer updates.

- Validation loss estimate (20 random batches): 10.9094 at step 0, 4.0805 at
  step 100, 3.7706 at step 200, and 3.5545 at step 300.
- No NaN or CUDA OOM occurred.
- Peak CUDA memory: 3899.2 MiB allocated, 5148.0 MiB reserved.
- Checkpoint: `out-tinystories/ckpt.pt`, 585.6 MiB, `iter_num=300`.
- Loading the checkpoint and continuing with updates 301 and 302 succeeded.

The fixed `max_iters` boundary now treats `max_iters` as the exact number of
completed optimizer updates. Evaluation and checkpoint I/O are excluded from
the following iteration's CPU wall-time measurement, but B-stage benchmarks
still need explicit CUDA synchronization or CUDA events.

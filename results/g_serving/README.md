# G-stage native inference

G1 establishes a versioned inference artifact before any KV-cache or serving
optimization is measured. Large exported weights and reference-logit tensors
live under `runs/checkpoints/g_serving/`; this directory contains only compact,
reviewable manifests and correctness results.

The first baseline uses the trained Dense TinyStories checkpoint at
`runs/checkpoints/legacy/out-tinystories/ckpt.pt` (8 layers, width 512,
8 heads, block size 512). CPU runs are correctness evidence only. Formal
latency and throughput results require a CUDA machine and are not inferred from
CPU timings.

G2 adds a preallocated contiguous KV cache for one-shot prefill and incremental
decode. The correctness matrix covers FP32/BF16, short/medium/long prompts,
batch sizes 1/2, and an exact-block-size prompt that forces explicit cache
rebuild after the learned-position sliding window shifts. G3 then measures the
native naive and KV-cache paths on one CUDA GPU.

## G1 result

Command:

```bash
python -m serving.export_checkpoint \
  --checkpoint=runs/checkpoints/legacy/out-tinystories/ckpt.pt
python -m serving.verify_export \
  --checkpoint=runs/checkpoints/legacy/out-tinystories/ckpt.pt \
  --artifact=runs/checkpoints/g_serving/dense_baseline/model.pt \
  --device=cpu
```

The 614,076,460-byte training checkpoint became a 204,690,819-byte inference
artifact. Optimizer, RNG and data-window state were excluded. A fresh process
strict-loaded the artifact twice, preserved tied embedding storage, and matched
the source checkpoint on 4/28/125-token prompts. FP32 and BF16 last-token logits
had zero difference and all four greedy continuation tokens matched. The
machine-readable evidence is in `g1_export/manifest.json` and
`g1_export/equivalence.json`.

## G2 result

Command:

```bash
python -m serving.verify_kv_cache \
  --artifact=runs/checkpoints/g_serving/dense_baseline/model.pt \
  --device=cpu
```

All 13 correctness workloads passed. FP32 maximum absolute logits difference
was `4.76837158203125e-06`; BF16 maximum was `0.03125`; every greedy token
matched full recomputation. The contiguous cache occupies 16 MiB per FP32
batch element or 8 MiB per BF16 batch element at the model's 512-token maximum.
An exact-512-token prompt correctly triggered one cache rebuild when generation
shifted the learned absolute-position window. See
`g2_kv_cache/correctness.json`.

The recorded CPU wall times are diagnostic only. No GPU latency or throughput
claim is made from these runs.

## G3 result

The full single-GPU matrix passed on an NVIDIA GeForce RTX 5060 Laptop GPU: 48
variants, three fresh processes per variant and 144 formal GPU processes in
total. It records CUDA-Event TTFT/TPOT, synchronized end-to-end latency,
throughput, peak CUDA memory, cache rebuilds, deterministic workload/output
hashes and source provenance.

With a 128-token prompt, KV caching was slower at batch 1, approximately broke
even at batch 8 with a long continuation, and delivered 1.831x--2.844x output
throughput at batch 32. With a 512-token prompt, learned absolute positions
forced a rebuild after every generated token; all KV/naive throughput ratios
were at or below 0.967x. Extra peak allocated memory ranged from 4.4 MiB at
batch 1 to 319.7 MiB at batch 32, depending on shape.

The 75%-shared-prefix cases are a baseline only: this native backend does not
yet reuse cache entries across requests. Full tables, limitations and
correctness diagnostics are in [`g3_native/README.md`](g3_native/README.md).
The frozen protocol and GPU commands are in
[`docs/g_stage_protocol.md`](../../docs/g_stage_protocol.md). The matrix can be
inspected without executing it:

```bash
python scripts/run_g3_native_matrix.py --gpu=0 --dry-run
```


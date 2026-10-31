# G3 native single-GPU serving benchmark

Date: 2026-10-17. Status: **pass**.

G3 compares full-context recomputation (`naive`) with the project's contiguous
KV cache (`kv_cache`) on one NVIDIA GeForce RTX 5060 Laptop GPU. The complete
frozen matrix contains 48 variants and three fresh processes per variant, for
144 formal GPU processes. Each process performs two warmups and five measured
repeats. The validated aggregate is [`summary.json`](summary.json); raw evidence
is under `raw/`.

## Workload and interpretation

- Artifact: the 50.91M-parameter Dense TinyStories model exported by G1.
- Precision: BF16.
- Batch sizes: 1, 8 and 32.
- Prompt/output lengths: 128 or 512 / 64 or 256 tokens.
- Prefix modes: no shared prefix and a 75%-shared-prefix workload.
- Decode inputs: fixed validation continuations, so both backends execute the
  same token sequence even when BF16 argmax is affected by a near tie.
- Aggregate: median of the three independent run medians.

The following table reports output tokens/s and the KV/naive ratio. Extra
memory is the difference in peak allocated CUDA memory.

| Prefix | Batch | Prompt | Output | Naive tok/s | KV tok/s | KV/naive | Extra peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1 | 128 | 64 | 179.3 | 148.9 | 0.830x | 6.9 MiB |
| none | 1 | 128 | 256 | 182.5 | 165.8 | 0.908x | 4.4 MiB |
| none | 8 | 128 | 64 | 1382.3 | 1311.6 | 0.949x | 56.9 MiB |
| none | 8 | 128 | 256 | 1186.2 | 1233.1 | 1.040x | 38.6 MiB |
| none | 32 | 128 | 64 | 2654.9 | 4962.4 | 1.869x | 227.8 MiB |
| none | 32 | 128 | 256 | 1708.8 | 4277.3 | 2.503x | 155.5 MiB |
| none | 1 | 512 | 64 | 190.0 | 142.9 | 0.752x | 9.5 MiB |
| none | 1 | 512 | 256 | 180.0 | 148.0 | 0.822x | 9.5 MiB |
| none | 8 | 512 | 64 | 765.5 | 739.9 | 0.967x | 79.5 MiB |
| none | 8 | 512 | 256 | 785.1 | 742.8 | 0.946x | 79.5 MiB |
| none | 32 | 512 | 64 | 878.1 | 828.0 | 0.943x | 319.6 MiB |
| none | 32 | 512 | 256 | 874.7 | 831.7 | 0.951x | 319.7 MiB |
| 75% shared | 1 | 128 | 64 | 180.3 | 155.0 | 0.860x | 6.9 MiB |
| 75% shared | 1 | 128 | 256 | 193.2 | 171.2 | 0.886x | 4.4 MiB |
| 75% shared | 8 | 128 | 64 | 1465.7 | 1281.5 | 0.874x | 56.9 MiB |
| 75% shared | 8 | 128 | 256 | 1175.0 | 1270.4 | 1.081x | 38.6 MiB |
| 75% shared | 32 | 128 | 64 | 2466.9 | 4515.8 | 1.831x | 227.8 MiB |
| 75% shared | 32 | 128 | 256 | 1674.6 | 4762.4 | 2.844x | 155.5 MiB |
| 75% shared | 1 | 512 | 64 | 183.4 | 164.5 | 0.897x | 9.5 MiB |
| 75% shared | 1 | 512 | 256 | 179.1 | 156.4 | 0.873x | 9.5 MiB |
| 75% shared | 8 | 512 | 64 | 813.8 | 721.9 | 0.887x | 79.5 MiB |
| 75% shared | 8 | 512 | 256 | 752.9 | 717.1 | 0.952x | 79.5 MiB |
| 75% shared | 32 | 512 | 64 | 886.4 | 824.3 | 0.930x | 319.6 MiB |
| 75% shared | 32 | 512 | 256 | 882.6 | 828.3 | 0.938x | 319.7 MiB |

## Findings

For a 128-token prompt, KV caching is not automatically beneficial on this
small model. It is slower at batch 1, reaches the crossover near batch 8 with a
256-token continuation, and is decisively useful at batch 32. The strongest
observed comparison is 2.844x output throughput for the 75%-shared-prefix,
batch-32, 256-output workload. KV caching also consumes extra memory that grows
with batch size.

The 512-token prompt already fills this model's learned absolute-position
window. Every subsequent token shifts all retained positions, so the cache
must be rebuilt. The 64- and 256-token workloads recorded exactly 63 and 255
rebuilds per measured repeat. All such comparisons are at or below 0.967x:
this is an explicit sliding-window failure mode, not evidence that incremental
KV caching is generally ineffective.

The native backend does not implement cross-request prefix reuse. Consequently,
the 75%-shared-prefix workload establishes a baseline but does not itself save
prefill work. A later prefix-aware runtime should be compared against these
same workload hashes.

## Numerical correctness note

FP32 greedy decoding matched for all 64 diagnosed steps, with maximum logits
difference `3.814697265625e-06`. BF16 first selected a different token at step
14 because one path produced logits `8.0` and `7.96875`, while the other rounded
both candidates to `8.0`. The timing matrix therefore feeds the same fixed
validation continuation to both implementations while retaining model-prediction
hashes separately. The diagnosis is preserved in
[`pilot_divergence.json`](pilot_divergence.json).

## Reproduction

```bash
python scripts/run_g3_native_matrix.py --gpu=0 --dry-run
python scripts/run_g3_native_matrix.py --gpu=0 --resume
python scripts/summarize_g3_serving.py \
  --raw-dir=results/g_serving/g3_native/raw \
  --output=results/g_serving/g3_native/summary.json
```

The experiment boundary and acceptance rules are frozen in
[`docs/g_stage_protocol.md`](../../../docs/g_stage_protocol.md).

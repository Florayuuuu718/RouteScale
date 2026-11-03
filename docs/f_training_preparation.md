# F complete-training preparation v1

Additional local scope frozen on 2026-11-03 after the module preparation. The
original [module protocol](f_stage_protocol.md) and acceptance stay unchanged.
No paid resources or multi-GPU performance are authorized by this document.

`ep_training.py` adds global admission for the fully replicated control,
shared-gradient SUMs, globally normalized accumulated loss and clipping.
The clipping norm counts each replicated parameter once and each owned expert
across ranks. Globally absent gradients remain absent. The independent audit
also reconstructs the clipped reference norm directly from raw gradients.

Local qualifiers use real Gloo2 and NCCL1. Every job runs the original single-
model reference, full-expert manual replication, and EP from identical seed1337.
The tiny scope uses two layers, width16, sequence8, global microbatch4,
two accumulated microbatches (64 global tokens), clip0.01 to force clipping,
and the original FP32 3e-5/3e-4 gate. CUDA tiny also uses the existing BF16
0.002/0.03 gate. The complete TinyStories scope uses the
[rental configuration](../config/f_ep_rental_requirements.json): all eight
layers/width512, global microbatch8, sequence512, four accumulations (16,384
tokens), BF16, four experts at layer4, global capacity1.25/drop, balance0.01,
clip1.0 and AdamW lr3e-4/betas0.9,0.95/decay0.1. Qualifications perform one
complete update and retain loss, sampled full-vocabulary logits, every parameter
gradient, updated weight, AdamW state and global window hashes. This is a real
TinyStories update, not a quality or throughput experiment. All three modes
share the same exact token windows. Unused experts retain None gradients.

The launcher holds the common GPU lock, rejects unavailable/busy GPUs, bounds
jobs and never overwrites artifacts. Full jobs allow 600s; benchmark jobs allow
900s, with process-group cleanup inherited from module preparation. Large raw
state tensors remain ignored under `runs/f_ep`; compact hashes/reports and
frozen sources belong in Git. CPU tiny and GPU qualifiers may be selected:

```bash
python3 scripts/prepare_f_training.py --output runs/f_ep/training_plan.json
python3 scripts/prepare_f_training.py --execute --output runs/f_ep/training001
```

The default local matrix is Gloo2 tiny FP32, NCCL1 tiny FP32/BF16 and NCCL1 full
BF16. `--host-qualification` additionally prepares real CUDA2 full BF16 on the
same authorized host. The full-model worker creates each model sequentially
and releases it before constructing the next, avoiding simultaneous training
graphs. All numerical audits precede performance.

`--benchmark --host-root PATH` prepares or executes nine processes: reference
CUDA1, replicated CUDA2 and EP CUDA2, each repeated three times. It requires
the prior eight module/GPT FP32/BF16 host qualifications and audited full-model
CUDA1/2 BF16 updates with matching source/raw hashes on this same host. CPU or
NCCL1-only local results cannot unlock it. Benchmark updates use 20 warmup,
100 measurement steps, the same deterministic window stream, no profiler and
no per-update logits/state capture. Timing includes CPU window assembly, H2D,
complete forward/backward, accumulation, synchronization, clipping and optimizer;
the per-rank maximum is reduced after each timed interval. No-profiler timing
does not establish communication/computation attribution; collect separate
NCCL traces in the authorized session. Native PyTorch DDP reducer timing remains
distinct from the explicit manual-reduction replicated control.

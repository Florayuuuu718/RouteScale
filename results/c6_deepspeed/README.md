# C6 DeepSpeed local smoke

`zero0_single_gpu` through `zero3_single_gpu` contain the resolved DeepSpeed
configuration and final result for each stage. Every stage completed a save
after update 1 and resumed in a new process to update 2. ZeRO-0 additionally
resumed from update 2 to update 4 for the native/FSDP2 trajectory comparison.

The checkpoint shards are ignored because they are generated binary
artifacts. Stage 1/2/3 on one rank exercise the implementation but cannot show
real partitioning. The ordinary 50.91M comparison keeps `zero.Init` disabled
so initialization matches the other stages. Capacity tests enable it
explicitly with `--zero-init` to avoid constructing a full model per rank.

`zero3_offload.json` is provided but was not run locally because this host has
no `nvcc`; offload remains an isolated four-GPU follow-up after ordinary
ZeRO-3 passes.

`zero2_single_gpu/fp32_verification.json` and
`zero3_single_gpu/fp32_verification.json` record successful consolidation to
ordinary FP32 state dictionaries and strict loading into a non-DeepSpeed
`GPT`. The generated `merged_fp32.pt` files are ignored.

Formal server measurements use `train_deepspeed.py --benchmark`. That mode
runs warmup and measured updates only, gathers per-rank CUDA Event timings and
memory, and deliberately skips validation and checkpoint I/O. It records the
resolved ZeRO configuration and the explicit fused-optimizer choice. See the
C-stage multi-GPU runbook for the 1/2/4-GPU, three-run matrix.

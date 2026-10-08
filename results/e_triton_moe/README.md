# E-stage fixed-capacity dispatch and Triton Grouped GEMM results

Date: 2026-10-08. E-stage replaces D's ragged per-expert dispatch with a
fixed-capacity layout, then compares PyTorch batched GEMM and a custom Triton
Grouped GEMM. It remains a single-GPU local Top-1 MoE experiment: there is no
Expert Parallel or All-to-All.

## Protocol

All formal runs used one RTX 5060 Laptop GPU, TinyStories full, BF16, an
8-layer/512-dimensional GPT, batch size 8, sequence length 512, four gradient
accumulation steps, and one four-expert MoE block at zero-based layer 4.
Capacity factor was 1.25, token drop was enabled, and balance-loss weight was
0.01. Every variant used 20 warmup plus 100 measured updates in three
independent processes.

The data-window hashes, workload, optimizer, initialization, common source
hashes, and MoE source hashes match across the matrix. Profiler and graph-break
diagnostics were separate from formal timing.

## Correctness

The fixed-capacity dispatcher assigns each token a deterministic expert-local
slot in a `[experts, capacity, hidden]` buffer. Overflow tokens use a
sentinel row and retain zero MoE output. Tests cover:

- loop versus padded output, expert counts, processed counts, and drop order;
- input, router, and expert parameter gradients for processed experts;
- Triton Grouped GEMM forward and backward against `torch.bmm`;
- BF16 padded-Triton MoE output and input gradients;
- the complete existing MoE regression suite.

The padded implementation evaluates all expert parameter groups. If an expert
receives no accepted token, it gets an explicit zero gradient instead of the
ragged loop's absent gradient; AdamW weight decay can therefore differ in that
edge case. The formal balanced workload used every expert.

## Grouped GEMM microbenchmark

| Shape | Python loop | `torch.bmm` | Triton grouped | Triton vs loop |
|---|---:|---:|---:|---:|
| `[4,1280,512]×[4,512,2048]` | 0.4730 ms | 0.7085 ms | 0.3149 ms | 1.502× |
| `[4,1280,2048]×[4,2048,512]` | 0.4230 ms | 0.3242 ms | 0.3202 ms | 1.321× |

Both Triton outputs matched `torch.bmm` with zero maximum absolute
difference for these BF16 inputs. This benchmark excludes routing, packing,
GELU, unpacking, backward, and the rest of GPT.

## Formal complete-update results

| Variant | Run medians (ms) | Final median | Throughput | Peak allocated |
|---|---|---:|---:|---:|
| Dense eager | 346.675 / 346.672 / 347.041 | 346.675 ms | 47,260 tok/s | 3,907.0 MiB |
| Dense compiled | 241.221 / 240.803 / 240.582 | 240.803 ms | 68,039 tok/s | 3,128.7 MiB |
| loop eager | 390.601 / 391.389 / 389.242 | 390.601 ms | 41,946 tok/s | 4,032.9 MiB |
| loop compiled | 400.400 / 398.550 / 399.095 | 399.095 ms | 41,053 tok/s | 4,826.6 MiB |
| padded PyTorch eager | 361.580 / 362.082 / 361.986 | 361.986 ms | 45,261 tok/s | 4,041.2 MiB |
| padded PyTorch compiled | 248.554 / 248.515 / 248.778 | 248.554 ms | 65,917 tok/s | 3,272.0 MiB |
| padded Triton eager | 360.571 / 360.710 / 360.486 | 360.571 ms | 45,439 tok/s | 4,042.4 MiB |
| padded Triton compiled | 248.545 / 248.384 / 248.451 | 248.451 ms | 65,945 tok/s | 3,273.4 MiB |

Key comparisons:

- fixed-capacity Triton eager is 1.083× faster than loop eager;
- compiled loop is 2.13% lower-throughput than loop eager;
- compiled padded Triton is 1.451× faster than its eager version;
- compiled padded Triton is 1.606× faster than compiled loop;
- compiled padded Triton reaches 96.92% of compiled Dense throughput and is
  only 3.08% lower;
- Triton is only 0.042% faster than padded `torch.bmm` in the
  complete compiled update, within practical run-to-run equivalence.

The result therefore separates two effects. Fixed-capacity layout and removal
of dynamic routing boundaries produced the large end-to-end gain. The custom
Grouped GEMM kernel won its isolated microbenchmark but added little over the
compiled padded PyTorch implementation at full-model scope.

## Profiler and compiler diagnosis

Across 10 active profiled updates, compiled loop → compiled padded Triton:

| Diagnostic | loop | padded Triton | Change |
|---|---:|---:|---:|
| CPU self time | 6,841 ms | 4,301 ms | -37.1% |
| CUDA self time | 3,453 ms | 2,544 ms | -26.3% |
| launch API calls | 21,977 | 15,710 | -28.5% |
| `copy_` calls | 1,240 | 280 | -77.4% |
| `nonzero` calls | 160 | 0 | -100% |
| `bincount` calls | 40 | 0 | -100% |
| `index_select` calls | 560 | 0 | -100% |
| `index_copy` calls | 160 | 0 | -100% |

The padded path introduced 40 `index_add` calls for packing and
recorded 164 grouped-kernel events. A separate
`TORCH_LOGS=graph_breaks,recompiles` smoke emitted no graph-break or
recompile record; D4's `bincount/nonzero` boundary was absent.

Profiler totals contain instrumentation overhead and are explanatory only.
Formal performance comes from the no-Profiler three-run matrix.

## Decision

The original hypothesis was partly correct:

1. Grouped GEMM can fuse expert GEMMs and is substantially faster than the
   Python GEMM loop in isolation.
2. The decisive end-to-end change is the fixed-capacity representation, because
   it removes dynamic shapes and lets the whole model benefit from compile.
3. On this four-expert workload, compiled `torch.bmm` and custom
   Triton are effectively tied. Triton should remain an experimental backend,
   not become the default solely from the 0.042% complete-update difference.

The next justified kernel work is not generic GEMM retuning. It is to measure
and reduce pack/unpack plus backward cost, then consider fused GELU/gating or
Triton backward only if those components are proven material.

Raw formal runs are under `benchmark/`, the microbenchmark is
`grouped_gemm.json`, the aggregate is `summary.json`,
Profiler evidence is under `profiler/`, and graph-break evidence is
under `diagnostics/`.

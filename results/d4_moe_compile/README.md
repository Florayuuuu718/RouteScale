# D4 `torch.compile` compatibility and performance

Date: 2026-10-08. D4 evaluates the compile-compatibility variable deliberately
left outside the D3 eager algorithm comparison. It does not add Expert Parallel,
All-to-All, or a new routing algorithm.

## Formal protocol

The D3 eager runs are the baseline. D4 adds three independent compiled Dense
runs and three independent compiled Top-1 MoE runs. Each process uses the same
RTX 5060 Laptop GPU, TinyStories full data, BF16, sequence length 512,
micro-batch 8, four gradient-accumulation steps, 16,384 tokens/update, 20 warmup
updates, and 100 measured updates. Profiler, evaluation, checkpointing, and
routing telemetry remain outside formal timing.

The eager and compiled groups have identical warmup and measurement window
hashes and matching `train.py`, `model.py`, and, for MoE, `moe.py` hashes. The
only intended model-execution change is `compile=False` to `compile=True`;
artifact names and output paths necessarily differ.

```bash
for run in run1 run2 run3; do
  uv run python train.py config/benchmark_d4_dense_compile.py \
    --benchmark_run_id="$run"
done

for run in run1 run2 run3; do
  uv run python train.py config/benchmark_d4_moe_compile.py \
    --benchmark_run_id="$run"
done

uv run python scripts/summarize_d4_moe_compile.py
```

## Formal results

| Variant | Mode | Run medians (ms/update) | Median | Tokens/s | Peak allocated | Peak reserved |
|---|---|---|---:|---:|---:|---:|
| Dense | eager | 347.281 / 347.165 / 347.190 | 347.190 ms | 47,190 | 3,907.0 MiB | 4,912 MiB |
| Dense | compiled | 240.432 / 240.737 / 240.013 | 240.432 ms | 68,144 | 3,128.7 MiB | 3,384 MiB |
| Top-1 MoE | eager | 391.070 / 384.411 / 391.047 | 391.047 ms | 41,898 | 4,032.8 MiB | 4,962 MiB |
| Top-1 MoE | compiled | 395.580 / 393.249 / 397.920 | 395.580 ms | 41,418 | 4,012.6 MiB | 4,930 MiB |

Dense gains 1.444x speed, or 44.40% throughput, while median allocated/reserved
memory decreases by 778.4/1,528 MiB. MoE reaches only 0.989x: step time rises
1.16% and throughput falls 1.15%, with a small 20.3/32 MiB memory reduction.
Consequently, compiled MoE reaches only 60.78% of compiled Dense throughput,
compared with 88.78% in eager mode.

All measured losses are finite. Because compiled BF16 kernels can choose
different reduction/fusion paths, training trajectories are not bitwise equal.
Across the 100 measured updates and three runs, the maximum absolute eager versus
compiled LM-loss difference is 0.0722 for Dense and 0.0926 for MoE. This is a
numerical-equivalence limit, not a failure to train; D3 remains the controlled
quality/stability comparison.

## Profiler and graph-break diagnosis

Two diagnostic runs use 20 unprofiled updates followed by `wait=2`, `warmup=2`,
and 10 active Profiler updates. Their timings contain instrumentation overhead
and are not substituted for the formal results.

| Diagnostic over 10 active updates | Eager MoE | Compiled MoE | Change |
|---|---:|---:|---:|
| Self CPU time | 5,534 ms | 5,811 ms | +5.0% |
| Self CUDA time | 3,456 ms | 3,432 ms | -0.7% |
| CUDA launch API calls | 24,271 | 21,960 | -9.5% |
| `aten::copy_` calls | 6,680 | 1,236 | -81.5% |
| `aten::mm` self-device time | 2,180.4 ms | 2,186.0 ms | +0.3% |
| `aten::nonzero` calls | 160 | 160 | unchanged |
| `aten::bincount` calls | 40 | 40 | unchanged |

The compiled trace contains 14 distinct `CompiledFxGraph` event names and 46
distinct Triton-kernel event names, so compilation did optimize substantial
dense portions of the model. It did not remove the data-dependent routing
boundary: `bincount` at `moe.py:153` and `nonzero` at `moe.py:176` caused
dynamic-shape graph breaks, the expert loop fell back across that boundary, and
different routed-token counts triggered expert-shape recompilation. The
unchanged routing-op counts and nearly unchanged matrix-multiply CUDA time
explain why copy fusion did not become an end-to-end MoE speedup.

Regenerate the diagnostic artifacts with:

```bash
uv run python train.py config/profile_d4_moe_eager.py
uv run python train.py config/profile_d4_moe_compile.py
uv run python scripts/summarize_d4_moe_profiler.py
```

The complete Chrome traces are intentionally stored under ignored
`logs/d4_moe_compile/` because they total roughly 233 MiB. Reviewable metadata,
key averages, the generated profiler summary, and a concise compiler-log record
are retained here.

## Decision

`torch.compile` remains beneficial for the Dense model but is not enabled for
the current Top-1 MoE by default. The next optimization target is the dispatch
representation itself: remove data-dependent `nonzero`/Python expert-loop graph
boundaries or adopt a grouped/padded expert implementation before attempting a
custom Triton kernel. D4 records this negative MoE performance result rather
than presenting Dense-only compile gains as an MoE optimization.

# B2 Profiler artifacts

The committed artifacts contain the small, reviewable summaries for two short
PyTorch Profiler runs:

- `baseline_*`: eager model (`compile=False`);
- `compiled_*`: the same workload with `compile=True`;
- `summary.json`: extracted trace comparison and invariant checks.

Both runs used 20 unprofiled startup updates followed by a Profiler schedule of
`wait=2, warmup=2, active=10`. Shapes and memory were recorded. The full Chrome
traces are intentionally kept in ignored `logs/b2_profiler/` because they are
109 MiB and 55 MiB.

Regenerate the traces:

```bash
uv run python train.py config/profile_tinystories.py
uv run python train.py config/profile_tinystories.py \
  --compile=True --profiler_run_id="compiled"
uv run python scripts/summarize_b2_profiler.py
```

Open either `*_trace.json` in Perfetto or another Chrome trace viewer. Profiler
timings include instrumentation overhead and must not be reported as formal
training throughput.

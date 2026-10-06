# B2 no-Profiler `torch.compile` A/B

The `eager/` and `compiled/` directories each contain three independent B1-style
runs plus an intra-group summary. `summary.json` validates the shared workload
and reports the cross-group comparison.

Only one intended performance variable changes:

```text
eager:    compile=False
compiled: compile=True
```

All six runs use the same model seed, data seed, measured-window hash, model,
batch, sequence length, BF16 precision, optimizer, 20 warmup updates and 100
measured updates. Rebuild the summaries with:

```bash
uv run python scripts/summarize_b1.py \
  --results-dir="results/b2_compile_ab/eager"
uv run python scripts/summarize_b1.py \
  --results-dir="results/b2_compile_ab/compiled"
uv run python scripts/compare_b2_compile.py
```

# B1 single-GPU benchmark artifacts

- `run1.json`, `run2.json`, `run3.json`: raw per-update CUDA Event timings,
  observed losses, environment, data manifest, source hashes and memory usage.
- `summary.json`: invariant checks and the median across the three independent
  process runs.

Regenerate the summary after producing exactly three run files:

```bash
uv run python scripts/summarize_b1.py
```

# Local run workspace

This directory contains mutable, machine-local outputs and is ignored by Git
except for this file.

- `checkpoints/`: model, optimizer and distributed checkpoint state;
- `traces/`: profiler traces and runtime logs;
- `archives/`: locally packaged evidence bundles;
- `tmp/`: disposable intermediate files created by run scripts.

Committed experiment metrics and reports belong in `results/`. Frozen evidence
copied from another machine belongs in `artifacts/`. New code must not write
checkpoints, profiler traces or large logs into either directory.

Historical JSON files keep the artifact paths recorded when those experiments
ran. Local legacy checkpoints have been moved under `checkpoints/legacy/` or
the matching `checkpoints/<stage>/` directory; the recorded JSON is not
rewritten because it is experiment evidence.

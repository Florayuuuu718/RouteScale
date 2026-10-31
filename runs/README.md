# Local run workspace

This directory contains mutable, machine-local outputs and is ignored by Git
except for this file.

- `checkpoints/`: model, optimizer and distributed checkpoint state;
- `traces/`: profiler traces and runtime logs;
- `archives/`: locally packaged evidence bundles;
- `tmp/`: disposable intermediate files created by run scripts.
- `vendor/`: pinned upstream source checkouts for isolated serving experiments;
- `envs/`: serving-specific Python environments;
- `toolchains/`: experiment-local compiler/SDK compatibility copies (I1 never edits system CUDA);
- `models/`: downloaded pretrained model snapshots;
- `downloads/`: verified third-party wheels and resumable download parts;
- `cache/`: experiment-specific compiler caches;
- `logs/`: experiment execution logs.
- `workloads/`: reproducible request token arrays (hashes are kept in `results/`);
- `validation/`: large numerical reference tensors used by correctness gates.

Committed experiment metrics and reports belong in `results/`. Frozen evidence
copied from another machine belongs in `artifacts/`. New code must not write
checkpoints, profiler traces or large logs into either directory.

Historical JSON files keep the artifact paths recorded when those experiments
ran. Local legacy checkpoints have been moved under `checkpoints/legacy/` or
the matching `checkpoints/<stage>/` directory; the recorded JSON is not
rewritten because it is experiment evidence.

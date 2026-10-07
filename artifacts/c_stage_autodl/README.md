# C-stage AutoDL evidence snapshot

This directory is the verified extraction of
`routescale_c_evidence_final_v2.tar.gz`, downloaded from the four-GPU AutoDL
host on 2026-10-07, plus the separately archived raw Chrome traces.

- Evidence archive SHA256: `13edff9130d3015b26e013044193347e2abae2a47a8a5365fd82cd1afbaa85fa`
- [Raw trace archive](routescale_c_raw_traces.tar.gz) SHA256: `30fef4572a60705f3c5e5346bfa30f259ff9b6f3cf892edda70b0918f30fc501`
- Manifest: [`artifact_manifest.json`](artifact_manifest.json)
- Included files: 360
- Included bytes before compression: 6,015,989
- Manifest verification: 360 passed, 0 failed
- Raw Chrome traces: 24 files; local overlap regeneration matched the stored
  summary byte for byte

The main evidence package intentionally contains small JSON, log, text,
Markdown and CSV evidence only. It excludes datasets, checkpoint payloads,
runtime directories and caches. Raw Chrome traces are kept in the separate
compressed archive, while derived summaries, per-rank key averages and trace
metadata remain directly browsable.

To inspect the timelines without polluting the working tree, extract the raw
archive at the root of a disposable checkout. Its `logs/` and `results/` paths
line up with the retained manifests; `bash scripts/run_c_trace_overlap.sh`
then regenerates `results/c_trace_overlap/summary.json`. Individual
`*_trace.json` files can be opened in Perfetto or Chrome tracing.

The main interpretation and links to individual evidence files are in
[`../../docs/c_stage_results.md`](../../docs/c_stage_results.md).

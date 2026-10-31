"""Recompute H3 metrics and verify the frozen numerical and file evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_performance import DEFAULT_CONFIG, VARIANTS, load_inputs, now
from scripts.summarize_h3_performance import summarize_runs, markdown_table


def audit(config_path, *, check_only=False):
    import torch
    from serving.h3_performance import compare_tensors
    config, _, cases, provenance = load_inputs(config_path)
    directory = ROOT / config["results_dir"]
    output = directory / "audit.json"
    if output.exists() and not check_only:
        raise ValueError("Preserve the completed audit; use --check-only")
    gate_path = directory / "gate.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "pass" or not gate["performance_execution_allowed"] or gate["provenance"] != provenance:
        raise ValueError("Numerical qualification missing or stale")
    baseline = json.loads((directory / "historical_baseline.json").read_text())["historical_json_sha256"]
    if any(digest(ROOT / name) != checksum for name, checksum in baseline.items()):
        raise ValueError("Historical G/H evidence changed")
    snapshot = json.loads((directory / "source_snapshot/manifest.json").read_text())["source_sha256"]
    for name, checksum in snapshot.items():
        if digest(directory / "source_snapshot" / name) != checksum or digest(ROOT / name) != checksum:
            raise ValueError("Source snapshot or current qualified source changed")
    pilot = json.loads((directory / "pilot_gate.json").read_text())
    if pilot["status"] != "pass" or pilot["qualification_gate_sha256"] != digest(gate_path):
        raise ValueError("Pilot qualification missing or stale")
    for relative, checksum in pilot["raw_sha256"].items():
        if digest(directory / relative) != checksum:
            raise ValueError("Pilot evidence changed")
    reports = {}
    for relative, checksum in gate["report_sha256"].items():
        path = directory / relative
        if digest(path) != checksum:
            raise ValueError("Qualification report changed")
        reports[relative] = json.loads(path.read_text())
    tensors = {}
    for name, report in reports.items():
        for case_id, metadata in report["cases"].items():
            path = ROOT / metadata["path"]
            if digest(path) != metadata["sha256"]:
                raise ValueError("Qualification tensor changed")
            tensors[name, case_id] = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    for case in cases:
        name = case["case_id"]
        for variant in VARIANTS:
            result = compare_tensors(tensors[f"qualification/{variant}.json", name],
                tensors["reference.json", name], config["tolerances"],
                fp32_reference=config.get("reference_policy") == "bf16_weights_fp32_math")
            if result != gate["comparisons"][f"{variant}/{name}"] or not result["passed"]:
                raise ValueError("Recomputed native comparison differs")
        result = compare_tensors(tensors["qualification/paged_graph.json", name],
            tensors["qualification/paged_eager.json", name], config["tolerances"], exact=True)
        if result != gate["comparisons"][f"graph_exact/{name}"] or not result["passed"]:
            raise ValueError("Recomputed Graph exactness differs")
    paths = sorted((directory / "raw").rglob("*.json"))
    runs = [(p, json.loads(p.read_text())) for p in paths]
    summary = summarize_runs(runs, config, cases, provenance, digest(gate_path))
    if check_only:
        if json.loads((directory / "summary.json").read_text()) != summary:
            raise ValueError("Saved summary differs from independent raw-metric recomputation")
    else:
        if (directory / "summary.json").exists():
            raise ValueError("Preserve existing summary")
        write_result(summary, directory / "summary.json")
        (directory / "tables.md").write_text(markdown_table(summary))
    record = dict(schema_version=1, stage="H3", status="pass", evidence_class="independent_performance_audit",
        created_at_utc=now(), provenance=provenance, qualification_gate_sha256=digest(gate_path),
        pilot_gate_sha256=digest(directory / "pilot_gate.json"),
        native_bf16_vectors_recompared=gate["native_bf16_vectors"],
        graph_vectors_recompared=gate["eager_graph_exact_vectors"],
        formal_processes=len(paths), raw_sha256={str(p.relative_to(directory)): digest(p) for p in paths},
        summary_sha256=digest(directory / "summary.json"), raw_metrics_recomputed_exactly=True,
        reference_tensor_hashes_match=True, old_correctness_scope_unchanged=True)
    if check_only:
        saved = json.loads(output.read_text())
        if {k: v for k, v in saved.items() if k != "created_at_utc"} != {k: v for k, v in record.items() if k != "created_at_utc"}:
            raise ValueError("Saved audit differs")
    else:
        write_result(record, output)
    print(f"H3 performance independent audit: pass, {len(paths)} formal processes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    audit(args.config, check_only=args.check_only)

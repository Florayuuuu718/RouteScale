"""Audit completed H2 Graph results against raw request records and current files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h2_runtime import aggregate_waves
from scripts.summarize_h2_nanovllm import summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/h_nanovllm/h2_graph_b8")
    args = parser.parse_args()
    directory = args.results_dir
    journal = json.loads((directory / "run.json").read_text())
    if journal["status"] != "pass" or journal["completed_formal_processes"] != journal["planned_processes"]:
        raise ValueError("Audit requires a completed pilot")
    gate = json.loads((directory / "preflight.json").read_text())
    gate_hash = digest(directory / "preflight.json")
    paths = sorted((directory / "raw").rglob("run*.json"))
    runs = [(path, json.loads(path.read_text())) for path in paths]
    regenerated = summarize_runs(runs, mechanism="cuda_graph")
    summary = json.loads((directory / "summary.json").read_text())
    if regenerated["variants"] != summary["variants"] or regenerated["comparisons"] != summary["comparisons"]:
        raise AssertionError("Saved summary differs from recomputed raw results")
    if len(paths) != journal["planned_processes"]:
        raise AssertionError("Unexpected or missing raw processes")
    outputs = {}
    total_requests = total_output_tokens = 0
    for path, run in runs:
        if run["provenance"] != gate["provenance"] or run["preflight_sha256"] != gate_hash:
            raise AssertionError("Raw process differs from its qualified preflight")
        if run["case_id"] not in gate["validated_case_ids"] or run["variant"] not in gate["allowed_variants"]:
            raise AssertionError("Raw process is outside its numerical scope")
        if run["compile_counters"]["before_measurement"] != run["compile_counters"]["after_measurement"]:
            raise AssertionError("Compilation contaminated timing")
        if any(run["released_state"].values()):
            raise AssertionError("Request/KV state leaked")
        for phase, waves in run["waves"].items():
            if len(waves) != run["workload"]["measured_pairs"] or aggregate_waves(waves) != run["phases"][phase]:
                raise AssertionError("Aggregate metrics do not match raw request timings")
            for pair, wave in enumerate(waves):
                for request in wave["requests"]:
                    n = request["output_tokens"]
                    if n != run["workload"]["output_length"] or request["cached_prompt_tokens"]:
                        raise AssertionError("Fixed-output or no-prefix contract changed")
                    expected_tpot = (request["e2e_ms"] - request["ttft_ms"]) / (n - 1)
                    if abs(expected_tpot - request["tpot_ms"]) > 1e-9:
                        raise AssertionError("TPOT formula differs from the documented contract")
                    key = (run["case_id"], run["run_id"], phase, pair, request["request_id"])
                    outputs.setdefault(key, {})[run["variant"]] = request["output_token_ids_sha256"]
                    total_requests += 1
                    total_output_tokens += n
    for name, checksum in gate["provenance"]["source_sha256"].items():
        if digest(ROOT / name) != checksum:
            raise AssertionError(f"Harness changed: {name}")
    weights = ROOT / "runs/models/Qwen3-0.6B/model.safetensors"
    if digest(weights) != gate["provenance"]["model_sha256"]:
        raise AssertionError("Model weights changed")
    if any(set(row) != set(gate["allowed_variants"]) for row in outputs.values()):
        raise AssertionError("Output comparison is missing a variant")
    report = {"schema_version": 1, "stage": "H2", "status": "pass", "scope": "cuda_graph_only",
        "evidence_class": "result_integrity_audit", "entrypoint_sha256": digest(Path(__file__)),
        "raw_processes": len(paths), "measured_requests": total_requests,
        "measured_output_tokens": total_output_tokens, "paired_output_hashes": len(outputs),
        "matching_output_hash_pairs": sum(len(set(row.values())) == 1 for row in outputs.values()),
        "raw_metrics_recomputed_exactly": True, "summary_recomputed_exactly": True,
        "current_harness_and_weights_match": True,
        "preflight_sha256": gate_hash,
        "raw_sha256": {str(path.relative_to(directory)): digest(path) for path in paths},
        "limitations": "Only the explicit batch-8 Graph scope. Matching sampled-token hashes are diagnostic, not a full-model quality evaluation."}
    write_result(report, directory / "audit.json")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_sha256"}, indent=2))


if __name__ == "__main__":
    main()

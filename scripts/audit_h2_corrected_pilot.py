"""Recompute corrected H2 metrics, qualification, scope, and paired token hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.benchmark_h2_corrected import (
    DEFAULT_PERFORMANCE, load_performance_inputs, performance_provenance, validate_timed_run,
)
from serving.h2_runtime import now
from scripts.run_h2_nanovllm import pilot_jobs
from scripts.summarize_h2_nanovllm import summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PERFORMANCE)
    args = parser.parse_args()
    inputs = load_performance_inputs(args.config)
    performance, _, pilot, config, _, cases, gate, gate_path = inputs
    directory = ROOT / performance["results_dir"]
    journal = json.loads((directory / "run.json").read_text())
    if journal["completed_formal_processes"] != 36 or journal["planned_processes"] != 36:
        raise ValueError("Audit requires all 36 formal processes")
    expected_provenance = performance_provenance(args.config, inputs)
    paths = sorted((directory / "raw").rglob("*.json"))
    runs = [(path, json.loads(path.read_text())) for path in paths]
    planned = set(pilot_jobs({**config, "pilot_case_ids": performance["pilot_case_ids"],
        "independent_runs": 3}, performance["variants"]))
    if len(runs) != 36 or {(r["case_id"], r["variant"], r["run_id"]) for _, r in runs} != planned:
        raise ValueError("Unexpected, duplicate, or missing raw processes")
    regenerated = summarize_runs(runs)
    summary = json.loads((directory / "summary.json").read_text())
    if any(regenerated[key] != summary[key] for key in ("scope", "variants", "comparisons", "variant_count")):
        raise ValueError("Saved summary differs from raw evidence")
    outputs, requests, tokens = {}, 0, 0
    for path, run in runs:
        validate_timed_run(run, cases)
        if (run["provenance"] != expected_provenance
                or run["qualification_gate_sha256"] != digest(gate_path)
                or run["engine"] != {**pilot["engine"], **config["variants"][run["variant"]], "dtype": config["dtype"]}
                or run["timing_policy"] != pilot["timing_policy"]
                or run["warmup_pairs"] != pilot["warmup_pairs"]
                or run["workload"]["measured_pairs"] != config["measured_pairs_per_run"]
                or run["workload"]["sampling"] != config["sampling"]
                or run["workload"]["seed"] != config["seed"]
                or run["capacity"]["actual_physical_blocks"] != pilot["engine"]["num_kvcache_blocks"]
                or run["capacity"]["kv_tensor_bytes"] != run["initialization"]["kv_tensor_bytes"]
                or run["free_bytes_after_run"] < pilot["minimum_free_after_kv_bytes"]):
            raise ValueError(f"Raw process differs from its qualified contract: {path}")
        for phase, waves in run["waves"].items():
            for pair, wave in enumerate(waves):
                for request in wave["requests"]:
                    key = (run["case_id"], run["run_id"], phase, pair, request["request_id"])
                    outputs.setdefault(key, {})[run["variant"]] = request["output_token_ids_sha256"]
                    requests += 1
                    tokens += request["output_tokens"]
    if any(set(row) != set(performance["variants"]) for row in outputs.values()):
        raise ValueError("Paired sampled outputs are missing a control")
    report = dict(schema_version=1, stage="H2", status="pass", evidence_class="result_integrity_audit",
        created_at_utc=now(), numerical_policy_id=gate["policy_id"], scope=performance["scope"],
        qualification_gate_sha256=digest(gate_path), raw_processes=len(paths), measured_requests=requests,
        measured_output_tokens=tokens, paired_request_sets=len(outputs),
        matching_sampled_outputs={name: sum(row[name] == row["eager_no_prefix"] for row in outputs.values())
            for name in ("graph_no_prefix", "eager_prefix")},
        raw_metrics_recomputed_exactly=True, summary_recomputed_exactly=True,
        current_harness_environment_and_weights_match=True, exact_prefix_hits_verified=True,
        raw_sha256={str(path.relative_to(directory)): digest(path) for path in paths},
        limitations="Only four fixed cases under this corrected policy. Sampled-token hashes are diagnostic, not a model-quality evaluation or universal numerical proof.")
    write_result(report, directory / "audit.json")
    print(json.dumps({k: v for k, v in report.items() if k != "raw_sha256"}, indent=2))


if __name__ == "__main__":
    main()

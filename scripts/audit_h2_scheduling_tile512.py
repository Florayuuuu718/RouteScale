"""Independently recompute saved scheduling results and verify historical lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h2_linear_policy import LINEAR_POLICY
from serving.h2_runtime import now
from serving.h2_scheduling import load_scheduling_inputs, scheduling_provenance
from scripts.summarize_h2_scheduling import check_gate, compare_probes, engine_contract, summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/serving/h2_scheduling_tile512.json")
    args = parser.parse_args()
    inputs = load_scheduling_inputs(args.config)
    config, _, pilot, base, _, cases, _ = inputs
    if config.get("linear_policy") != LINEAR_POLICY:
        raise ValueError("This audit qualifies only the explicit tile512 strategy")
    directory = ROOT / config["results_dir"]
    journal = json.loads((directory / "run.json").read_text())
    if journal["status"] != "pass" or journal["completed_formal_processes"] != 18:
        raise ValueError("Independent verification requires the completed 18-process pilot")
    gate_path = directory / "preflight.json"
    gate = json.loads(gate_path.read_text())
    check_gate(gate, scheduling_provenance(args.config, inputs), engine_contract(pilot, config), cases)
    reports = {policy: json.loads((directory / "preflight" / f"{policy}.json").read_text()) for policy in config["policies"]}
    if compare_probes(reports, config, pilot, cases) != gate["comparisons"]:
        raise ValueError("Saved numerical gate differs from complete probe records")
    for policy, checksum in gate["report_sha256"].items():
        if digest(directory / "preflight" / f"{policy}.json") != checksum:
            raise ValueError("Numerical report changed")
    for case in reports["fixed_cohort"]["cases"].values():
        reference = case["reference_logits"]
        if digest(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError("Numerical reference tensor changed")
    paths = sorted((directory / "raw").rglob("*.json"))
    runs = [(path, json.loads(path.read_text())) for path in paths]
    regenerated = summarize_runs(runs, config, pilot, base, cases, gate)
    summary = json.loads((directory / "summary.json").read_text())
    if any(regenerated[key] != summary[key] for key in ("variants", "comparisons", "engine", "provenance")):
        raise ValueError("Saved summary differs from independently recomputed raw records")
    metadata = dict(**LINEAR_POLICY, linear_modules=112, padding_rows=0,
        weight_dtype_unchanged=True, decode_unchanged=True)
    outputs, uniform = {}, {}
    for path, run in runs:
        if (run["qualification_gate_sha256"] != digest(gate_path)
                or run["initialization"]["prefill_linear_policy"] != metadata):
            raise ValueError("Gate binding or complete-model adapter installation changed")
        for index, trace in enumerate(run["traces"]):
            for row in trace["requests"]:
                key = (run["case_id"], run["run_id"], index, row["request_id"])
                outputs.setdefault(key, {})[run["policy"]] = row["output_token_ids_sha256"]
            if run["case_id"] == "equal_burst":
                key = (run["run_id"], index)
                uniform.setdefault(key, {})[run["policy"]] = [
                    (step["is_prefill"], step["request_ids"]) for step in trace["steps"]]
    if any(set(row) != set(config["policies"]) for row in outputs.values()):
        raise ValueError("Paired request evidence is incomplete")
    if any(row["fixed_cohort"] != row["continuous"] for row in uniform.values()):
        raise ValueError("Equal-length negative-control step structure differs")
    audit = json.loads((directory / "audit.json").read_text())
    actual_hashes = {str(p.relative_to(directory)): digest(p) for p in paths}
    if actual_hashes != audit["raw_sha256"]:
        raise ValueError("Raw result hashes differ from the completed audit")
    historical = {}
    for name in ("h2_graph_b8", "h2_corrected_pilot"):
        old = ROOT / "results/h_nanovllm" / name
        hashes = json.loads((old / "audit.json").read_text())["raw_sha256"]
        for relative, checksum in hashes.items():
            if digest(old / relative) != checksum:
                raise ValueError("A historical formal result changed")
        historical[name] = len(hashes)
    old = ROOT / config["previous_failed_results"]
    snapshot = json.loads((old / "source_snapshot/manifest.json").read_text())
    failed_gate = json.loads((old / "preflight.json").read_text())
    if failed_gate["status"] != "fail" or digest(old / "preflight.json") != snapshot["numerical_gate_sha256"]:
        raise ValueError("Original failed numerical gate changed")
    for name, checksum in failed_gate["provenance"]["scheduling_source_sha256"].items():
        if snapshot["files"][name]["sha256"] != checksum:
            raise ValueError("Archived harness differs from original failure provenance")
    for policy, checksum in failed_gate["report_sha256"].items():
        if digest(old / "preflight" / f"{policy}.json") != checksum:
            raise ValueError("Original failure report changed")
    report = dict(schema_version=1, stage="H2", status="pass", evidence_class="independent_scheduling_verification",
        created_at_utc=now(), audit_entrypoint_sha256=digest(Path(__file__)),
        raw_processes=len(runs), all_112_linear_adapters_verified=True,
        full_vocabulary_vector_comparisons=sum(r["compared_steps"] for r in gate["comparisons"]),
        all_numerical_steps_exact=True, summary_and_raw_metrics_recomputed_exactly=True,
        source_model_environment_and_reference_hashes_match=True,
        paired_request_sets=len(outputs), matching_sampled_output_hashes=sum(len(set(r.values())) == 1 for r in outputs.values()),
        sampled_hashes_by_case={case: dict(paired=sum(key[0] == case for key in outputs),
            matching=sum(key[0] == case and len(set(row.values())) == 1 for key, row in outputs.items()))
            for case in cases},
        equal_burst_step_structure_identical=True,
        historical_formal_raw_hashes_match=historical, original_failure_and_source_snapshot_match=True,
        outline_git_diff_empty=not subprocess.check_output(["git", "diff", "--", "docs/RouteScale_实施教程大纲.md"], cwd=ROOT, text=True),
        qualification_gate_sha256=digest(gate_path), raw_sha256=actual_hashes,
        limitations="Only the explicit BF16 tile512 runtime and finite 16-request traces. Sampled hashes across different policies are diagnostic because sampler RNG consumption differs. Not native upstream timing, model quality or online SLO evidence.")
    write_result(report, directory / "verification.json")
    print(json.dumps({k: v for k, v in report.items() if k != "raw_sha256"}, indent=2))


if __name__ == "__main__":
    main()

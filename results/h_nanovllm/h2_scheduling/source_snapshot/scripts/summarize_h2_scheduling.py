"""Build a cross-admission numerical gate; recompute and audit all timed records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h2_runtime import now
from serving.h2_scheduling import (
    DEFAULT_SCHEDULING, POLICIES, SWITCHES, load_scheduling_inputs,
    scheduling_provenance, summarize_traces, validate_trace,
)


def engine_contract(pilot, config):
    return {**pilot["engine"], **SWITCHES, "dtype": "bfloat16", "max_active_requests": config["max_active_requests"]}


def compare_probes(reports, config, pilot, cases):
    if set(reports) != set(POLICIES):
        raise ValueError("Both independent policy probes are required")
    baseline = reports["fixed_cohort"]
    for policy, report in reports.items():
        if (report.get("status") != "pass" or report.get("evidence_class") != "functional_gpu_scheduling_probe"
                or report.get("policy") != policy or report["provenance"] != baseline["provenance"]
                or report["engine"] != engine_contract(pilot, config) or set(report["cases"]) != set(cases)
                or report["kv_tensor_bytes"] <= 0 or report["kv_tensor_bytes"] != baseline["kv_tensor_bytes"]
                or report["compiler_settings"] != dict(recompile_limit=64, fail_on_recompile_limit_hit=True, suppress_errors=False)
                or report["free_bytes_after_run"] < pilot["minimum_free_after_kv_bytes"]):
            raise ValueError("Incomplete, failed, or mismatched numerical probes")
        for case_id, case in cases.items():
            result = report["cases"][case_id]
            if (result["workload_sha256"] != case["workload_sha256"] or not result["pass"]
                    or not result["functional_trace"]["finite_logits_checked"]):
                raise ValueError("Input stream, finite-logit check, or numerical result changed")
            validate_trace(result["functional_trace"], case, config, policy, timing=False)
    comparisons = []
    for case_id, case in cases.items():
        rows = reports["continuous"]["cases"][case_id]["comparisons"]
        if set(rows) != {r["request_id"] for r in case["requests"]}:
            raise ValueError("Numerical comparison is missing a request")
        for request in case["requests"]:
            row = rows[request["request_id"]]
            if (row["steps"] != request["output_tokens"] or not row["allclose"]
                    or not math.isfinite(row["max_relative_rmse"])
                    or row["max_relative_rmse"] > pilot["correctness_max_relative_rmse"]):
                raise ValueError("Cross-policy logits exceed the unchanged numerical budget")
        comparisons.append(dict(case_id=case_id, workload_sha256=case["workload_sha256"],
            compared_steps=sum(r["steps"] for r in rows.values()),
            exact_steps=sum(r["exact_steps"] for r in rows.values()),
            max_absolute_error=max(r["max_absolute_error"] for r in rows.values()),
            max_relative_rmse=max(r["max_relative_rmse"] for r in rows.values()), pass_numerical=True))
    return comparisons


def check_gate(gate, provenance, engine, cases):
    expected = {key: sum(r["output_tokens"] for r in case["requests"]) for key, case in cases.items()}
    if (gate.get("status") != "pass" or gate.get("gpu_execution_allowed") is not True
            or gate.get("evidence_class") != "scheduling_numerical_gate"
            or gate.get("tolerances") != dict(atol=.05, rtol=.02, max_relative_rmse=.005)
            or gate["provenance"] != provenance or gate["engine"] != engine
            or gate["allowed_policies"] != list(POLICIES)
            or set(gate["case_ids"]) != set(cases)):
        raise ValueError("Scheduling gate is failed, stale or outside this policy/engine scope")
    rows = gate["comparisons"]
    if (len(rows) != len(cases) or {r["case_id"] for r in rows} != set(cases)
            or any(r["compared_steps"] != expected[r["case_id"]] or r["pass_numerical"] is not True
                or r["workload_sha256"] != cases[r["case_id"]]["workload_sha256"]
                or not math.isfinite(r["max_relative_rmse"]) or r["max_relative_rmse"] > .005
                or not math.isfinite(r["max_absolute_error"])
                or not 0 <= r["exact_steps"] <= r["compared_steps"] for r in rows)):
        raise ValueError("Gate has incomplete numerical coverage")


def write_gate(path, inputs):
    config, _, pilot, _, _, cases, _ = inputs
    directory = ROOT / config["results_dir"]
    provenance = scheduling_provenance(path, inputs)
    reports = {name: json.loads((directory / "preflight" / f"{name}.json").read_text()) for name in POLICIES}
    report = dict(schema_version=1, stage="H2", evidence_class="scheduling_numerical_gate", status="fail",
        gpu_execution_allowed=False, created_at_utc=now(), provenance=provenance,
        engine=engine_contract(pilot, config), allowed_policies=list(POLICIES), case_ids=list(cases),
        kv_tensor_bytes=reports["fixed_cohort"]["kv_tensor_bytes"],
        tolerances=dict(atol=pilot["correctness_atol"], rtol=pilot["correctness_rtol"],
            max_relative_rmse=pilot["correctness_max_relative_rmse"]),
        clock="logical 10 ms per model step for deterministic validation arrivals, not performance",
        report_sha256={name: digest(directory / "preflight" / f"{name}.json") for name in POLICIES})
    try:
        report["comparisons"] = compare_probes(reports, config, pilot, cases)
        if any(r["provenance"] != provenance for r in reports.values()):
            raise ValueError("Current model/environment/source differs from the probes")
        for metadata in reports["fixed_cohort"]["cases"].values():
            reference = metadata["reference_logits"]
            if digest(ROOT / reference["path"]) != reference["sha256"]:
                raise ValueError("Numerical reference tensors changed")
        report.update(status="pass", gpu_execution_allowed=True)
    except (ValueError, KeyError) as error:
        report["failure"] = str(error)
    write_result(report, directory / "preflight.json")
    check_gate(report, provenance, engine_contract(pilot, config), cases)
    return report


def validate_run(run, config, pilot, base, cases, gate):
    if (run.get("schema_version") != 1 or run.get("stage") != "H2"
            or run.get("status") != "pass" or run.get("evidence_class") != "formal_gpu_scheduling"
            or run.get("scope") != "admission_policy" or run.get("policy") not in POLICIES
            or run.get("run_id") not in ("run1", "run2", "run3") or run.get("case_id") not in cases):
        raise ValueError("Summary accepts only passed, bounded formal scheduling processes")
    check_gate(gate, run["provenance"], run["engine"], cases)
    if (run["engine"] != engine_contract(pilot, config)
            or run["workload_sha256"] != cases[run["case_id"]]["workload_sha256"]
            or run["timing_policy"] != config["timing_policy"] or run["sampling"] != base["sampling"]
            or run["seed"] != base["seed"] or run["warmup_traces"] != 1 or run["measured_traces"] != 3
            or run["kv_tensor_bytes"] != gate["kv_tensor_bytes"]
            or run["compiler_settings"] != dict(recompile_limit=64, fail_on_recompile_limit_hit=True, suppress_errors=False)
            or run["compile_counters"]["before"] != run["compile_counters"]["after"]
            or run["free_bytes_after_run"] < pilot["minimum_free_after_kv_bytes"]
            or len(run["traces"]) != config["measured_traces"]):
        raise ValueError("Formal process changed the common timing, compiler or engine contract")
    for trace in run["traces"]:
        validate_trace(trace, cases[run["case_id"]], config, run["policy"])
        if trace["finite_logits_checked"]:
            raise ValueError("Logit probing contaminated formal timing")
    if run["metrics"] != summarize_traces(run["traces"]):
        raise ValueError("Saved metrics differ from raw per-request timing")


def summarize_runs(runs, config, pilot, base, cases, gate):
    if len(runs) != len(cases) * len(POLICIES) * config["independent_runs"]:
        raise ValueError("Expected all 18 independent scheduling processes")
    groups = {}
    for path, run in runs:
        validate_run(run, config, pilot, base, cases, gate)
        groups.setdefault((run["case_id"], run["policy"]), []).append((path, run))
    if set(groups) != {(case, policy) for case in cases for policy in POLICIES}:
        raise ValueError("Missing scheduling control")
    rows, sides = [], {}
    for (case, policy), members in sorted(groups.items()):
        members.sort(key=lambda pair: pair[1]["run_id"])
        if [r["run_id"] for _, r in members] != ["run1", "run2", "run3"]:
            raise ValueError("Missing or duplicated independent run IDs")
        row = dict(case_id=case, policy=policy,
            metrics={key: median(r["metrics"][key] for _, r in members) for key in members[0][1]["metrics"]},
            run_files=[str(path) for path, _ in members])
        rows.append(row)
        sides.setdefault(case, {})[policy] = row["metrics"]
    comparisons = []
    for case, policies in sorted(sides.items()):
        fixed, continuous = policies["fixed_cohort"], policies["continuous"]
        comparisons.append(dict(case_id=case,
            output_throughput_ratio=continuous["output_tokens_per_second"] / fixed["output_tokens_per_second"],
            ttft_p95_speedup=fixed["ttft_p95_ms"] / continuous["ttft_p95_ms"],
            e2e_p95_speedup=fixed["e2e_p95_ms"] / continuous["e2e_p95_ms"],
            extra_itl_max_ms=continuous["itl_max_ms"] - fixed["itl_max_ms"]))
    return dict(schema_version=1, stage="H2", status="pass", scope="admission_policy", created_at_utc=now(),
        aggregation="median of three independent process statistics, not global online tail latency",
        variants=rows, comparisons=comparisons, provenance=gate["provenance"], engine=gate["engine"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SCHEDULING)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    inputs = load_scheduling_inputs(args.config)
    config, _, pilot, base, _, cases, _ = inputs
    if args.gate_only:
        report = write_gate(args.config, inputs)
        print("Scheduling gate:",report["status"])
        return
    directory = ROOT / config["results_dir"]
    gate_path = directory / "preflight.json"
    gate = json.loads(gate_path.read_text())
    check_gate(gate, scheduling_provenance(args.config, inputs), engine_contract(pilot, config), cases)
    for name, checksum in gate["report_sha256"].items():
        if digest(directory / "preflight" / f"{name}.json") != checksum:
            raise ValueError("Numerical probe report changed")
    paths = sorted((directory / "raw").rglob("*.json"))
    runs = [(path, json.loads(path.read_text())) for path in paths]
    summary = summarize_runs(runs, config, pilot, base, cases, gate)
    hashes, requests, tokens = {}, 0, 0
    for path, run in runs:
        if run["qualification_gate_sha256"] != digest(gate_path):
            raise ValueError("Raw process uses a different gate")
        for index, trace in enumerate(run["traces"]):
            for row in trace["requests"]:
                key = (run["case_id"], run["run_id"], index, row["request_id"])
                hashes.setdefault(key, {})[run["policy"]] = row["output_token_ids_sha256"]
                requests += 1
                tokens += row["output_tokens"]
    if any(set(row) != set(POLICIES) for row in hashes.values()):
        raise ValueError("Missing paired output records")
    audit = dict(schema_version=1, stage="H2", status="pass", evidence_class="scheduling_integrity_audit",
        raw_processes=len(paths), measured_requests=requests, measured_output_tokens=tokens,
        paired_request_sets=len(hashes), matching_sampled_output_hashes=sum(len(set(row.values())) == 1 for row in hashes.values()),
        raw_metrics_recomputed_exactly=True, current_sources_model_environment_match=True,
        arrivals_budgets_active_cap_and_release_verified=True,
        qualification_gate_sha256=digest(gate_path), created_at_utc=now(),
        raw_sha256={str(p.relative_to(directory)): digest(p) for p in paths},
        limitations="Cross-policy sampler RNG consumption differs; output hashes are diagnostic, not a numerical gate or quality score. Fixed-cohort batches compact finished requests rather than deliberately computing padding.")
    write_result(summary, directory / "summary.json")
    write_result(audit, directory / "audit.json")
    print("Scheduling summary and audit: pass;",requests,"requests,",tokens,"output tokens")


if __name__ == "__main__":
    main()

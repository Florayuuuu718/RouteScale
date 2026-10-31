"""Fail-closed KV-pressure gate, measured summary and saved-logit/history audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h2_kv_pressure import (
    DEFAULT_CONFIG, POLICIES, engine_contract, load_inputs, summarize_pressure_traces, validate_pressure_trace,
)
from serving.h2_runtime import now
from serving.h2_linear_policy import LINEAR_POLICY

COMPILER = dict(recompile_limit=64, fail_on_recompile_limit_hit=True, suppress_errors=False)


def check_report(report, config, case, provenance, pilot, base, *, probe):
    expected_class = "functional_gpu_kv_pressure" if probe else "formal_gpu_kv_pressure"
    if (report["status"] != "pass" or report["evidence_class"] != expected_class
            or report["scope"] != "kv_funded_admission" or report["case_id"] != case["case_id"]
            or report["policy"] not in POLICIES or report["provenance"] != provenance
            or report["engine"] != engine_contract(pilot, case) or report["workload_sha256"] != case["workload_sha256"]
            or report["kv_tensor_bytes"] != case["num_blocks"]*29360128
            or report["compiler_settings"] != COMPILER or report["free_bytes_after_run"] < pilot["minimum_free_after_kv_bytes"]
            or report["sampling"] != base["sampling"] or report["seed"] != base["seed"]
            or report["timing_policy"] != config["timing_policy"] or report["admission_policy"] != config["admission_policy"]
            or report["warmup_traces"] != 1 or report["measured_traces"] != 3):
        raise ValueError("GPU evidence class, source, engine, budget, sampling or compiler contract changed")
    if report["initialization"]["prefill_linear_policy"] != dict(**LINEAR_POLICY,
            linear_modules=112, padding_rows=0, weight_dtype_unchanged=True, decode_unchanged=True):
        raise ValueError("The complete shared BF16 linear control was not installed")
    if probe:
        numerical = report["numerical"]
        if (not numerical["passed"] or not numerical["finite_logits_all_steps"] or not numerical["kv_poisoned_before_run"]
                or numerical["full_vocabulary_vectors"] != 16*case["output_tokens"]
                or numerical["saved_logits"]["shape"] != [16, case["output_tokens"], 151936]
                or numerical["saved_logits"]["dtype"] != "bfloat16"
                or set(numerical["counters"]) != {r["request_id"] for r in case["requests"]}):
            raise ValueError("Incomplete full-vocabulary numerical coverage")
        for row in numerical["counters"].values():
            if row["steps"] != case["output_tokens"]:
                raise ValueError("Missing teacher-forced output step")
            if report["policy"] == "paged" and (row["exact_steps"] != case["output_tokens"]
                    or row["max_absolute_error"] != 0 or row["max_relative_rmse"] != 0):
                raise ValueError("Every full-vocabulary vector must be exact")
        if not numerical["functional_trace"]["finite_logits_checked"]:
            raise ValueError("Poisoned-cache finite-logit validation missing")
        validate_pressure_trace(numerical["functional_trace"], case, config, report["policy"])
    else:
        if report["compile_counters"]["before"] != report["compile_counters"]["after"] or len(report["traces"]) != 3:
            raise ValueError("Measurement compilation or repetition count changed")
        for trace in report["traces"]:
            if trace["finite_logits_checked"]:
                raise ValueError("Numerical teacher-forced data cannot enter measured results")
            validate_pressure_trace(trace, case, config, report["policy"])
        if report["metrics"] != summarize_pressure_traces(report["traces"]):
            raise ValueError("Saved performance metrics did not recompute exactly")


def gate_contract(config, cases, provenance, pilot):
    return dict(provenance=provenance, admission_policy=config["admission_policy"],
        policies=list(POLICIES), require_exact_logits=True,
        cases={key: dict(engine=engine_contract(pilot, case), workload_sha256=case["workload_sha256"],
            full_vocabulary_vectors=16*case["output_tokens"]) for key,case in cases.items()})


def check_gate(gate, config, cases, provenance, pilot):
    contract = gate_contract(config, cases, provenance, pilot)
    if (gate["status"] != "pass" or gate["evidence_class"] != "kv_pressure_numerical_gate"
            or not gate["gpu_performance_execution_allowed"] or gate["contract"] != contract
            or gate["compared_vectors"] != 4160 or gate["exact_vectors"] != 4160
            or set(gate["report_sha256"]) != {f"preflight/{key}_{p}.json" for key in cases for p in POLICIES}):
        raise ValueError("KV-pressure performance requires its own complete exact numerical gate")


def qualify(directory, config, cases, provenance, pilot, base):
    report = dict(schema_version=1, stage="H2", evidence_class="kv_pressure_numerical_gate", status="running",
        created_at_utc=now(), gpu_performance_execution_allowed=False,
        contract=gate_contract(config, cases, provenance, pilot), report_sha256={}, compared_vectors=0, exact_vectors=0)
    try:
        for key, case in cases.items():
            for policy in POLICIES:
                path = directory / "preflight" / f"{key}_{policy}.json"
                data = json.loads(path.read_text())
                check_report(data, config, case, provenance, pilot, base, probe=True)
                metadata = data["numerical"]["saved_logits"]
                if digest(ROOT / metadata["path"]) != metadata["sha256"]:
                    raise ValueError("Numerical tensor changed before qualification")
                report["report_sha256"][str(path.relative_to(directory))] = digest(path)
                if policy == "paged":
                    report["compared_vectors"] += sum(r["steps"] for r in data["numerical"]["counters"].values())
                    report["exact_vectors"] += sum(r["exact_steps"] for r in data["numerical"]["counters"].values())
        report.update(status="pass", gpu_performance_execution_allowed=True)
        check_gate(report, config, cases, provenance, pilot)
    except Exception as error:
        report.update(status="fail", gpu_performance_execution_allowed=False, failure=str(error))
    return report


def summarize_runs(runs, config, cases, provenance, pilot, base, gate, gate_sha):
    check_gate(gate, config, cases, provenance, pilot)
    expected = {(case, policy, f"run{i}") for case in cases for policy in POLICIES for i in (1,2,3)}
    found, groups = set(), {}
    for path, run in runs:
        key = (run["case_id"], run["policy"], run["run_id"])
        if key not in expected or key in found or run["qualification_gate_sha256"] != gate_sha:
            raise ValueError("Missing/duplicate/foreign formal process or stale gate")
        found.add(key)
        check_report(run, config, cases[key[0]], provenance, pilot, base, probe=False)
        groups.setdefault(key[:2], []).append(run)
    if found != expected:
        raise ValueError("Missing formal process; never summarize an incomplete matrix")
    variants = []
    for (case, policy), rows in sorted(groups.items()):
        metrics = {key: statistics.median(r["metrics"][key] for r in rows) for key in rows[0]["metrics"]}
        variants.append(dict(case_id=case, policy=policy, independent_runs=3, metrics=metrics,
            kv_tensor_bytes=rows[0]["kv_tensor_bytes"],
            initialization_seconds=statistics.median(r["initialization"]["initialization_seconds"] for r in rows)))
    pairs = []
    for case in cases:
        rows = {v["policy"]: v["metrics"] for v in variants if v["case_id"] == case}
        left, right = rows["static_window"], rows["paged"]
        pairs.append(dict(case_id=case, output_throughput_ratio=right["output_tokens_per_second"]/left["output_tokens_per_second"],
            ttft_p95_ratio=right["ttft_p95_ms"]/left["ttft_p95_ms"], e2e_p95_ratio=right["e2e_p95_ms"]/left["e2e_p95_ms"],
            tpot_p50_ratio=right["tpot_p50_ms"]/left["tpot_p50_ms"],
            resident_requests=[left["max_resident_requests"],right["max_resident_requests"]],
            peak_used_blocks=[left["peak_used_kv_blocks"],right["peak_used_kv_blocks"]]))
    return dict(schema_version=1, stage="H2", status="pass", evidence_class="formal_kv_pressure_summary",
        provenance=provenance, raw_processes=18, measured_requests=864, measured_output_tokens=74880,
        variants=variants, comparisons=pairs, limitations=config["limitations"])


def audit(directory, config, inputs, cases, provenance):
    import torch
    pilot, base = inputs[2], inputs[3]
    gate_path = directory / "preflight.json"
    gate = json.loads(gate_path.read_text())
    check_gate(gate, config, cases, provenance, pilot)
    for name, checksum in gate["report_sha256"].items():
        if digest(directory / name) != checksum:
            raise ValueError("Numerical report changed after qualification")
    compared = 0
    for key, case in cases.items():
        tensors = []
        for policy in POLICIES:
            data = json.loads((directory / "preflight" / f"{key}_{policy}.json").read_text())
            check_report(data, config, case, provenance, pilot, base, probe=True)
            metadata = data["numerical"]["saved_logits"]
            if digest(ROOT / metadata["path"]) != metadata["sha256"]:
                raise ValueError("Saved full-vocabulary tensor changed")
            tensor = torch.load(ROOT / metadata["path"], weights_only=True, mmap=True, map_location="cpu")
            if tensor.shape != (16, case["output_tokens"], 151936) or tensor.dtype != torch.bfloat16:
                raise ValueError("Saved tensor shape/dtype mismatch")
            tensors.append(tensor)
        for left,right in zip(*tensors):
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                raise ValueError("Saved numerical tensor has nonfinite logits")
            compared += int((left == right).all(dim=1).sum())
    if compared != 4160:
        raise ValueError("Independent saved-logit comparison was not fully exact")
    paths = sorted((directory / "raw").rglob("*.json"))
    runs = [(p,json.loads(p.read_text())) for p in paths]
    summary = summarize_runs(runs, config, cases, provenance, pilot, base, gate, digest(gate_path))
    saved = directory / "summary.json"
    if saved.exists() and json.loads(saved.read_text()) != summary:
        raise ValueError("Saved measured summary differs from full raw recomputation")
    historical = {}
    for name in ("h2_graph_b8", "h2_corrected_pilot", "h2_scheduling_tile512"):
        old = ROOT / "results/h_nanovllm" / name
        hashes = json.loads((old / "audit.json").read_text())["raw_sha256"]
        for relative, checksum in hashes.items():
            if digest(old / relative) != checksum:
                raise ValueError("Historical formal raw evidence changed")
        historical[name] = len(hashes)
    # Equal-budget negative control must execute the same request-step sequence.
    structure, outputs = {}, {}
    for path, run in runs:
        for index, trace in enumerate(run["traces"]):
            if run["case_id"] == "ample_short":
                structure.setdefault((run["run_id"],index), {})[run["policy"]] = [
                    (s["is_prefill"],s["request_ids"]) for s in trace["steps"]]
            for request in trace["requests"]:
                key = (run["case_id"],run["run_id"],index,request["request_id"])
                outputs.setdefault(key,{})[run["policy"]] = request["output_token_ids_sha256"]
    if any(row["static_window"] != row["paged"] for row in structure.values()):
        raise ValueError("Ample-budget negative-control model call structure differs")
    if subprocess.check_output(["git", "diff", "--", "docs/RouteScale_实施教程大纲.md"],cwd=ROOT,text=True):
        raise ValueError("Unexpected implementation-outline edit")
    verification = dict(schema_version=1, stage="H2", status="pass", evidence_class="kv_pressure_result_audit",
        created_at_utc=now(), provenance=provenance, gate_sha256=digest(gate_path),
        raw_processes=len(runs), full_vocabulary_vectors_recompared=compared,
        summary_and_raw_metrics_recomputed_exactly=True, ample_control_step_structure_identical=True,
        historical_formal_raw_hashes_match=historical, outline_git_diff_empty=True,
        sampled_hash_pairs=len(outputs), matching_sampled_hash_pairs=sum(len(set(row.values()))==1 for row in outputs.values()),
        sampled_hashes_by_case={case: dict(paired=sum(k[0]==case for k in outputs),
            matching=sum(k[0]==case and len(set(v.values()))==1 for k,v in outputs.items())) for case in cases},
        raw_sha256={str(p.relative_to(directory)):digest(p) for p in paths}, limitations=config["limitations"])
    return summary, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, inputs, cases, provenance = load_inputs(args.config)
    directory = ROOT / config["results_dir"]
    if args.preflight:
        output = directory / "preflight.json"
        if output.exists():
            raise ValueError("Never overwrite a qualification gate")
        gate = qualify(directory, config, cases, provenance, inputs[2], inputs[3])
        write_result(gate, output)
        print(f"KV-pressure gate: {gate['status']}, {gate['exact_vectors']}/{gate['compared_vectors']} exact",flush=True)
        if gate["status"] != "pass":
            raise SystemExit(1)
    else:
        summary, verification = audit(directory, config, inputs, cases, provenance)
        if args.check_only:
            saved = json.loads((directory / "audit.json").read_text())
            if saved["raw_sha256"] != verification["raw_sha256"] or saved["provenance"] != provenance:
                raise ValueError("Completed audit hashes changed")
        else:
            if (directory / "audit.json").exists():
                raise ValueError("Use --check-only to preserve an existing audit")
            write_result(summary, directory / "summary.json")
            write_result(verification, directory / "audit.json")
        print(f"KV-pressure audit: pass, {verification['raw_processes']} formal processes, {verification['full_vocabulary_vectors_recompared']} exact vectors",flush=True)


if __name__ == "__main__":
    main()

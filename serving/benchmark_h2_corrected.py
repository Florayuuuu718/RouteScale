"""Bounded performance pilot for the separately qualified corrected H2 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_corrected_runtime import (
    close_corrected_engine, corrected_provenance, create_corrected_engine,
    load_corrected_inputs,
)
from serving.h2_runtime import (
    aggregate_waves, assert_released, check_capacity, compile_counts, execute_wave,
    now, reset_cache,
)
from serving.verify_h2_corrected import compare_reports
from scripts.summarize_h2_nanovllm import validate_run

DEFAULT_PERFORMANCE = ROOT / "config/serving/h2_corrected_pilot.json"
PERFORMANCE_SOURCES = (
    "serving/benchmark_h2_corrected.py", "scripts/run_h2_corrected_pilot.py",
    "scripts/audit_h2_corrected_pilot.py",
)


def validate_qualification(performance, validation, pilot, config, cases, gate):
    """Pure, fail-closed checks; never turn a historical native gate into a pass."""
    if (gate.get("status") != "pass" or gate.get("gpu_execution_allowed") is not True
            or gate.get("evidence_class") != "corrected_policy_numerical_gate"
            or gate.get("policy_id") != validation["policy_id"]
            or gate.get("policy") != validation["policy"]
            or gate.get("common_engine") != pilot["engine"]):
        raise ValueError("Corrected-policy qualification is missing, failed, or different")
    names = ["eager_no_prefix", "graph_no_prefix", "eager_prefix"]
    if (performance["variants"] != names or set(gate["allowed_variants"]) != set(names)
            or performance["independent_runs"] != 3
            or performance["scope"] != "cuda_graph_and_prefix_cache"):
        raise ValueError("Only the bounded three-variant, three-run pilot is authorized")
    if performance["pilot_case_ids"] != config["pilot_case_ids"]:
        raise ValueError("Performance scope must retain the original four pilot cases")
    if not set(performance["pilot_case_ids"]) <= set(gate["validated_case_ids"]):
        raise ValueError("A performance case lacks numerical qualification")
    expected = {(case_id, variant, phase) for case_id in gate["validated_case_ids"]
        for variant in names[1:] for phase in ("cold", "warm")}
    rows = gate.get("comparisons", [])
    if (len(rows) != len(expected) or {(r["case_id"], r["variant"], r["phase"]) for r in rows} != expected
            or any(r.get("exact") is not True or r.get("compared_steps") != 64
                or r.get("mismatched_steps") != [] for r in rows)
            or gate.get("compared_logit_tensors") != len(expected) * 64):
        raise ValueError("Qualification requires complete, exact 64-step comparisons")
    if config["measured_pairs_per_run"] != 5 or pilot["warmup_pairs"] != 1:
        raise ValueError("The qualified pilot fixes five measured pairs and one warmup pair")
    for case_id in performance["pilot_case_ids"]:
        check_capacity(pilot["engine"], cases[case_id])


def load_performance_inputs(path=DEFAULT_PERFORMANCE):
    performance = json.loads(path.read_text())
    validation_path = ROOT / performance["validation_config"]
    validation, pilot, config, h1, prepared = load_corrected_inputs(validation_path)
    cases = {case["case_id"]: case for case in prepared}
    directory = ROOT / validation["results_dir"]
    gate_path = directory / "gate.json"
    if digest(gate_path) != performance["qualification_gate_sha256"]:
        raise ValueError("Pinned numerical gate changed; preserve evidence and requalify separately")
    gate = json.loads(gate_path.read_text())
    validate_qualification(performance, validation, pilot, config, cases, gate)
    if gate["provenance"]["validation_config_sha256"] != digest(validation_path):
        raise ValueError("Numerical validation configuration changed")
    for field in ("source_sha256", "corrected_source_sha256"):
        for name, checksum in gate["provenance"][field].items():
            if digest(ROOT / name) != checksum:
                raise ValueError(f"Qualified source changed: {name}")
    reports = {name: json.loads((directory / f"{name}.json").read_text())
        for name in performance["variants"]}
    if compare_reports(reports, gate["validated_case_ids"]) != gate["comparisons"]:
        raise ValueError("Numerical reports no longer reproduce the pinned gate")
    for report in reports.values():
        if report["provenance"] != gate["provenance"]:
            raise ValueError("Numerical report provenance differs from the gate")
        for case_id, case in cases.items():
            if report["cases"][case_id]["workload_sha256"] != case["workload_sha256"]:
                raise ValueError("Numerical input stream changed")
    return performance, validation, pilot, config, h1, cases, gate, gate_path


def performance_provenance(path, inputs):
    performance, validation, pilot, config, h1, _, gate, _ = inputs
    current = corrected_provenance(ROOT / performance["validation_config"], validation, pilot, config, h1)
    if current != gate["provenance"]:
        raise ValueError("Current model, environment, or qualified runtime differs from the gate")
    return {**current, "performance_config_sha256": digest(path),
        "performance_source_sha256": {name: digest(ROOT / name) for name in PERFORMANCE_SOURCES}}


def validate_timed_run(run, cases):
    validate_run(run)
    if run.get("compiler_settings") != dict(recompile_limit=64,
            fail_on_recompile_limit_hit=True, suppress_errors=False):
        raise ValueError("Compiler must fail rather than silently fall back")
    if run["compile_counters"]["before_measurement"] != run["compile_counters"]["after_measurement"]:
        raise ValueError("Compilation contaminated measurement")
    if any(run["released_state"].values()):
        raise ValueError("Request or KV state leaked")
    case = cases[run["case_id"]]
    if run["workload"]["sha256"] != case["workload_sha256"]:
        raise ValueError("Timed workload differs from its numerical qualification")
    for phase in ("cold", "warm"):
        waves = run["waves"][phase]
        expected_hit = case["expected_warm_cached_tokens_per_request"] if phase == "warm" and run["engine"]["prefix_cache"] else 0
        if len(waves) != run["workload"]["measured_pairs"] or aggregate_waves(waves) != run["phases"][phase]:
            raise ValueError("Metrics differ from raw request timings")
        expected_ids = [r["request_id"] for r in case["waves"][phase]]
        for wave in waves:
            if ([r["request_id"] for r in wave["requests"]] != expected_ids
                    or wave["steps"] != case["output_length"]
                    or wave["cached_prompt_tokens"] != case["batch_size"] * expected_hit):
                raise ValueError("Timed wave coverage or cache hits changed")
            for request in wave["requests"]:
                if (request["output_tokens"] != case["output_length"]
                        or request["prompt_tokens"] != case["prompt_length"]
                        or request["cached_prompt_tokens"] != expected_hit
                        or len(request["output_token_ids_sha256"]) != 64):
                    raise ValueError("Request budget or exact prefix-hit contract changed")
                expected_tpot = (request["e2e_ms"] - request["ttft_ms"]) / (request["output_tokens"] - 1)
                if abs(expected_tpot - request["tpot_ms"]) > 1e-9:
                    raise ValueError("TPOT formula changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        raise SystemExit("Do not overwrite raw evidence; use a fresh result path")
    report = dict(schema_version=1, stage="H2", status="running", evidence_class="formal_gpu",
        case_id=args.case_id, variant=args.variant, run_id=args.run_id, created_at_utc=now())
    phase, llm = "qualification", None
    try:
        inputs = load_performance_inputs(args.config)
        performance, validation, pilot, config, h1, cases, _, gate_path = inputs
        if (args.case_id not in performance["pilot_case_ids"] or args.variant not in performance["variants"]
                or args.run_id not in ("run1", "run2", "run3")):
            raise ValueError("Requested process is outside the bounded pilot")
        case, switches = cases[args.case_id], config["variants"][args.variant]
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA device")
        torch.manual_seed(config["seed"])
        report["provenance"] = performance_provenance(args.config, inputs)
        report["qualification_gate_sha256"] = digest(gate_path)
        report["engine"] = {**pilot["engine"], **switches, "dtype": config["dtype"]}
        report["workload"] = {"sha256": case["workload_sha256"],
            **{k: case[k] for k in ("batch_size", "prompt_length", "output_length", "prefix_mode")},
            "measured_pairs": config["measured_pairs_per_run"], "sampling": config["sampling"],
            "seed": config["seed"], "cache_reset_policy": config["cache_reset_policy"]}
        report["timing_policy"] = pilot["timing_policy"]
        phase = "engine_initialization"
        llm, report["initialization"] = create_corrected_engine(h1, pilot, switches, validation)
        actual = llm.model_runner.kv_cache.size(2)
        if actual != pilot["engine"]["num_kvcache_blocks"]:
            raise AssertionError("Physical GPU KV pool changed")
        report["capacity"] = {"actual_physical_blocks": actual,
            "cold_decode_required_blocks": check_capacity(pilot["engine"], case),
            "kv_tensor_bytes": llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()}
        phase = "excluded_workload_warmup"
        started = time.perf_counter()
        reset_cache(llm)
        for name in ("cold", "warm"):
            execute_wave(llm, case, name, config)
        torch.cuda.synchronize()
        report.update(workload_warmup_seconds=time.perf_counter() - started, warmup_pairs=1)
        phase = "measurements"
        before = compile_counts()
        waves = {"cold": [], "warm": []}
        for index in range(config["measured_pairs_per_run"]):
            reset_cache(llm)
            torch.manual_seed(config["seed"] + index)
            for name in waves:
                waves[name].append(execute_wave(llm, case, name, config)[0])
        report["compile_counters"] = {"before_measurement": before, "after_measurement": compile_counts()}
        report["compiler_settings"] = dict(recompile_limit=torch._dynamo.config.recompile_limit,
            fail_on_recompile_limit_hit=torch._dynamo.config.fail_on_recompile_limit_hit,
            suppress_errors=torch._dynamo.config.suppress_errors)
        report["waves"] = waves
        report["phases"] = {name: aggregate_waves(rows) for name, rows in waves.items()}
        assert_released(llm)
        report["released_state"] = dict(running=0, waiting=0, used_blocks=0, nonzero_refcounts=0)
        report["free_bytes_after_run"] = torch.cuda.mem_get_info()[0]
        if report["free_bytes_after_run"] < pilot["minimum_free_after_kv_bytes"]:
            raise RuntimeError("GPU headroom dropped below the safety margin")
        validate_timed_run({**report, "status": "pass"}, cases)
        report["status"] = "pass"
    except BaseException:
        report.update(status="fail", failure_phase=phase, traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                close_corrected_engine(llm)
        except BaseException:
            report.update(status="fail", failure_phase="engine_cleanup", traceback=traceback.format_exc())
            raise
        finally:
            report["finished_at_utc"] = now()
            write_result(report, args.result)
    print(f"Corrected H2: {args.case_id}/{args.variant}/{args.run_id} passed", flush=True)


if __name__ == "__main__":
    main()

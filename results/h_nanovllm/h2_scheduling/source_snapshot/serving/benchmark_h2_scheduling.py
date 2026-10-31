"""An isolated numerical probe or timed admission-policy process on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_corrected_runtime import create_corrected_engine, close_corrected_engine
from serving.h2_runtime import assert_released, compile_counts, now
from serving.h2_scheduling import (
    DEFAULT_SCHEDULING, POLICIES, SWITCHES, execute_trace, load_scheduling_inputs,
    scheduling_provenance, summarize_traces, validate_trace,
)


def numerical_case(llm, case, config, pilot, base, policy, reference_report):
    import torch
    buffers = {r["request_id"]: [] for r in case["requests"]}
    compared = {r["request_id"]: dict(steps=0, exact_steps=0, allclose=True,
        max_absolute_error=0.0, max_relative_rmse=0.0, top1_equal_steps=0) for r in case["requests"]}
    reference = None
    path = ROOT / "runs/validation/h2_scheduling" / Path(config["results_dir"]).name / f"{case['case_id']}_fixed.pt"
    if policy == "continuous":
        metadata = reference_report["cases"][case["case_id"]]["reference_logits"]
        path = ROOT / metadata["path"]
        if digest(path) != metadata["sha256"]:
            raise ValueError("Reference logit tensor changed")
        reference = torch.load(path, map_location="cpu", weights_only=True)
    elif path.exists():
        raise ValueError("Preserve existing numerical tensors; use a fresh results_dir")

    def probe(request, step, logits):
        key = request["request_id"]
        if reference is None:
            if len(buffers[key]) != step:
                raise ValueError("Reference token order changed")
            buffers[key].append(logits.clone())
            return
        expected = reference[key][step]
        a, b = logits.float(), expected.float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite cross-policy logit")
        difference = a - b
        relative = difference.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)
        row = compared[key]
        if row["steps"] != step:
            raise ValueError("Candidate token order changed")
        row["steps"] += 1
        row["exact_steps"] += int(torch.equal(logits, expected))
        row["allclose"] &= torch.allclose(a, b, atol=pilot["correctness_atol"], rtol=pilot["correctness_rtol"])
        row["max_absolute_error"] = max(row["max_absolute_error"], difference.abs().max().item())
        row["max_relative_rmse"] = max(row["max_relative_rmse"], relative.item())
        row["top1_equal_steps"] += int(a.argmax() == b.argmax())

    trace = execute_trace(llm, case, config, base, policy, probe=probe)
    validate_trace(trace, case, config, policy, timing=False)
    result = dict(workload_sha256=case["workload_sha256"], functional_trace=trace,
        clock="logical 10 ms per model step; not a latency benchmark")
    if reference is None:
        if any(len(buffers[r["request_id"]]) != r["output_tokens"] for r in case["requests"]):
            raise ValueError("Reference output budget incomplete")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(buffers, path)
        result["reference_logits"] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), dtype="bfloat16")
        result["pass"] = True
    else:
        result["comparisons"] = compared
        result["pass"] = all(compared[r["request_id"]]["steps"] == r["output_tokens"]
            and compared[r["request_id"]]["allclose"]
            and compared[r["request_id"]]["max_relative_rmse"] <= pilot["correctness_max_relative_rmse"]
            for r in case["requests"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SCHEDULING)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--run-id")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        raise SystemExit("Never overwrite raw or numerical evidence")
    report = dict(schema_version=1, stage="H2", scope="admission_policy", status="running",
        evidence_class="functional_gpu_scheduling_probe" if args.probe else "formal_gpu_scheduling",
        policy=args.policy, created_at_utc=now())
    llm, phase = None, "inputs"
    try:
        inputs = load_scheduling_inputs(args.config)
        config, validation, pilot, base, h1, cases, _ = inputs
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU")
        torch.manual_seed(base["seed"])
        report["provenance"] = scheduling_provenance(args.config, inputs)
        report["engine"] = {**pilot["engine"], **SWITCHES, "dtype": "bfloat16", "max_active_requests": config["max_active_requests"]}
        report["timing_policy"] = config["timing_policy"]
        report.update(sampling=base["sampling"], seed=base["seed"],
            warmup_traces=config["warmup_traces"], measured_traces=config["measured_traces"])
        directory = ROOT / config["results_dir"]
        reference = None
        if args.probe and args.policy == "continuous":
            reference = json.loads((directory / "preflight/fixed_cohort.json").read_text())
            if reference["status"] != "pass" or reference["provenance"] != report["provenance"]:
                raise ValueError("Reference numerical process is missing or stale")
        if not args.probe:
            if args.case_id not in cases or args.run_id not in ("run1", "run2", "run3"):
                raise ValueError("Formal process is outside the bounded scheduling scope")
            from scripts.summarize_h2_scheduling import check_gate
            gate_path = directory / "preflight.json"
            check_gate(json.loads(gate_path.read_text()), report["provenance"], report["engine"], cases)
            report.update(case_id=args.case_id, run_id=args.run_id,
                qualification_gate_sha256=digest(gate_path), workload_sha256=cases[args.case_id]["workload_sha256"])
        phase = "engine_initialization"
        llm, report["initialization"] = create_corrected_engine(h1, pilot, SWITCHES, validation)
        if llm.model_runner.kv_cache.size(2) != pilot["engine"]["num_kvcache_blocks"]:
            raise ValueError("Physical KV pool changed")
        report["kv_tensor_bytes"] = llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()
        if args.probe:
            phase = "cross_policy_logit_validation"
            report["cases"] = {}
            for case in cases.values():
                report["cases"][case["case_id"]] = numerical_case(llm, case, config, pilot, base, args.policy, reference)
                write_result(report, args.result)
                print(args.policy, case["case_id"], report["cases"][case["case_id"]]["pass"], flush=True)
            report["status"] = "pass" if all(c["pass"] for c in report["cases"].values()) else "fail"
        else:
            phase = "excluded_workload_warmup"
            case = cases[args.case_id]
            started = time.perf_counter()
            execute_trace(llm, case, config, base, args.policy)
            report["workload_warmup_seconds"] = time.perf_counter() - started
            phase = "measurements"
            before = compile_counts()
            report["traces"] = []
            for index in range(config["measured_traces"]):
                torch.manual_seed(base["seed"] + index)
                trace = execute_trace(llm, case, config, base, args.policy)
                validate_trace(trace, case, config, args.policy)
                report["traces"].append(trace)
            report["compile_counters"] = dict(before=before, after=compile_counts())
            if before != report["compile_counters"]["after"]:
                raise ValueError("Compilation occurred inside measurement")
            report["metrics"] = summarize_traces(report["traces"])
            report["status"] = "pass"
        assert_released(llm)
        report["compiler_settings"] = dict(recompile_limit=torch._dynamo.config.recompile_limit,
            fail_on_recompile_limit_hit=torch._dynamo.config.fail_on_recompile_limit_hit,
            suppress_errors=torch._dynamo.config.suppress_errors)
        report["free_bytes_after_run"] = torch.cuda.mem_get_info()[0]
        if report["free_bytes_after_run"] < pilot["minimum_free_after_kv_bytes"]:
            raise ValueError("GPU safety headroom violated")
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
    print(f"Scheduling {args.policy}: {report['status']} -> {args.result}", flush=True)


if __name__ == "__main__":
    main()

"""One independent GPU process: numerical qualification OR measured KV pressure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_kv_pressure import (
    DEFAULT_CONFIG, POLICIES, create_engine, engine_contract, execute_trace, load_inputs,
    summarize_pressure_traces, validate_pressure_trace,
)
from serving.h2_runtime import assert_released, compile_counts, now
from serving.h2_scheduling import close_scheduling_engine


def numerical_case(llm, config, case, base, policy, provenance):
    import torch
    path = ROOT / "runs/validation/h2_kv_pressure" / Path(config["results_dir"]).name / f"{case['case_id']}_{policy}.pt"
    if path.exists():
        raise ValueError("Preserve existing saved numerical tensors")
    count = case["output_tokens"]
    buffer = torch.empty((16, count, 151936), dtype=torch.bfloat16, device="cpu")
    counters = {r["request_id"]: dict(steps=0, exact_steps=0, max_absolute_error=0.0,
        max_relative_rmse=0.0) for r in case["requests"]}
    indices = {r["request_id"]: i for i, r in enumerate(case["requests"])}
    reference, baseline = None, None
    if policy == "paged":
        report_path = ROOT / config["results_dir"] / "preflight" / f"{case['case_id']}_static_window.json"
        baseline = json.loads(report_path.read_text())
        if (baseline["status"] != "pass" or baseline["provenance"] != provenance
                or baseline["workload_sha256"] != case["workload_sha256"]):
            raise ValueError("Baseline numerical qualification is missing or stale")
        metadata = baseline["numerical"]["saved_logits"]
        if digest(ROOT / metadata["path"]) != metadata["sha256"]:
            raise ValueError("Reference tensor checksum changed")
        reference = torch.load(ROOT / metadata["path"], map_location="cpu", weights_only=True, mmap=True)
        if reference.shape != buffer.shape or reference.dtype != buffer.dtype:
            raise ValueError("Reference tensor shape/dtype changed")

    def probe(request, step, logits):
        key = request["request_id"]
        row = counters[key]
        if row["steps"] != step or logits.shape != (151936,):
            raise ValueError("Full-vocabulary vector or teacher token order changed")
        buffer[indices[key], step].copy_(logits)
        row["steps"] += 1
        if reference is not None:
            expected = reference[indices[key], step]
            difference = logits.float()-expected.float()
            row["exact_steps"] += int(torch.equal(logits, expected))
            row["max_absolute_error"] = max(row["max_absolute_error"], float(difference.abs().max()))
            relative = difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-12)
            row["max_relative_rmse"] = max(row["max_relative_rmse"], float(relative))

    trace = execute_trace(llm, case, config, base, policy, probe=probe)
    validate_pressure_trace(trace, case, config, policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buffer, path)
    passed = all(r["steps"] == count and (reference is None or r["exact_steps"] == count) for r in counters.values())
    return dict(passed=passed, saved_logits=dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
        shape=list(buffer.shape), dtype="bfloat16"), counters=counters, functional_trace=trace,
        full_vocabulary_vectors=16*count, finite_logits_all_steps=True, kv_poisoned_before_run=True,
        limitations="Teacher-forced probe only; its CPU copies and timings are never used as performance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        raise SystemExit("Never overwrite numerical or formal raw evidence")
    report = dict(schema_version=1, stage="H2", scope="kv_funded_admission", status="running",
        evidence_class="functional_gpu_kv_pressure" if args.probe else "formal_gpu_kv_pressure",
        case_id=args.case_id, policy=args.policy, created_at_utc=now())
    llm, phase = None, "inputs"
    try:
        config, inputs, cases, provenance = load_inputs(args.config)
        case = cases[args.case_id]
        _, _, pilot, base, _, _, _ = inputs
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU")
        torch.manual_seed(base["seed"])
        report.update(provenance=provenance, engine=engine_contract(pilot, case),
            workload_sha256=case["workload_sha256"], timing_policy=config["timing_policy"],
            sampling=base["sampling"], seed=base["seed"], warmup_traces=config["warmup_traces"],
            measured_traces=config["measured_traces"], admission_policy=config["admission_policy"])
        if not args.probe:
            if args.run_id not in ("run1", "run2", "run3"):
                raise ValueError("Unknown formal repetition")
            from scripts.summarize_h2_kv_pressure import check_gate
            gate_path = ROOT / config["results_dir"] / "preflight.json"
            check_gate(json.loads(gate_path.read_text()), config, cases, provenance, pilot)
            report.update(run_id=args.run_id, qualification_gate_sha256=digest(gate_path))
        phase = "engine_initialization"
        llm, report["initialization"] = create_engine(inputs, case)
        pool = llm.model_runner.kv_cache
        report["kv_tensor_bytes"] = pool.numel()*pool.element_size()
        if report["kv_tensor_bytes"] != case["num_blocks"]*29360128 or pool.size(2) != case["num_blocks"]:
            raise ValueError("Actual physical KV budget differs from declared case")
        if args.probe:
            phase = "strict_numerical_gate"
            report["numerical"] = numerical_case(llm, config, case, base, args.policy, provenance)
            report["status"] = "pass" if report["numerical"]["passed"] else "fail"
        else:
            phase = "excluded_workload_warmup"
            started = time.perf_counter()
            execute_trace(llm, case, config, base, args.policy)
            report["workload_warmup_seconds"] = time.perf_counter()-started
            before = compile_counts()
            phase, report["traces"] = "measurements", []
            for index in range(config["measured_traces"]):
                torch.manual_seed(base["seed"]+index)
                trace = execute_trace(llm, case, config, base, args.policy)
                validate_pressure_trace(trace, case, config, args.policy)
                report["traces"].append(trace)
            report["compile_counters"] = dict(before=before, after=compile_counts())
            if before != report["compile_counters"]["after"]:
                raise ValueError("Compilation inside the measurement window")
            report.update(status="pass", metrics=summarize_pressure_traces(report["traces"]))
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
                close_scheduling_engine(llm)
        except BaseException:
            report.update(status="fail", failure_phase="engine_cleanup", traceback=traceback.format_exc())
            raise
        finally:
            report["finished_at_utc"] = now()
            write_result(report, args.result)
    print(f"KV pressure {args.case_id}/{args.policy}: {report['status']}", flush=True)


if __name__ == "__main__":
    main()

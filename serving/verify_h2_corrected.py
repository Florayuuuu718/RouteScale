"""Validate corrected H2 paths with streaming, full-vocabulary BF16 logit hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from serving.backends.nanovllm import ROOT, write_result
from serving.h2_corrected_runtime import (
    DEFAULT_VALIDATION, close_corrected_engine, configure_compiler, corrected_provenance,
    create_corrected_engine, load_corrected_inputs,
)
from serving.h2_runtime import assert_released, compile_counts, now, reset_cache
from serving.h2_workloads import token_hash


def verify_wave(llm, case, phase, config):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context
    requests = case["waves"][phase]
    assert_released(llm)
    params = SamplingParams(**config["sampling"], max_tokens=64)
    for request in requests:
        llm.add_request(request["prompt_token_ids"], params)
    sequences = list(llm.scheduler.waiting)
    rows, cached = [], []
    while not llm.is_finished():
        step = len(rows)
        seqs, is_prefill = llm.scheduler.schedule()
        if len(seqs) != case["batch_size"] or is_prefill != (step == 0):
            raise AssertionError("Validation requires one full Prefill and no preemption")
        if is_prefill:
            cached = [seq.num_cached_tokens for seq in seqs]
        runner = llm.model_runner
        inputs, positions = runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
        try:
            with torch.inference_mode():
                logits = runner.run_model(inputs, positions, is_prefill).detach().cpu().contiguous()
            if logits.dtype != torch.bfloat16 or not torch.isfinite(logits).all():
                raise AssertionError("Expected finite native BF16 logits")
            checksum = hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest()
            rows.append({"step": step, "is_prefill": is_prefill, "shape": list(logits.shape),
                "dtype": "bfloat16", "sha256": checksum, "top1_token_ids": logits.argmax(-1).tolist()})
        finally:
            reset_context()
        # The same fixed continuation is supplied to every variant; no sampler
        # variation can conceal or amplify an earlier numerical difference.
        llm.scheduler.postprocess(seqs, [seq.prompt_token_ids[-1-step] for seq in seqs], is_prefill)
    expected = case["expected_warm_cached_tokens_per_request"] if llm._h2_prefix_cache and phase == "warm" else 0
    if len(rows) != 64 or cached != [expected] * case["batch_size"]:
        raise AssertionError("Output budget or prefix hit count changed")
    assert_released(llm)
    return {"logits": rows, "cached_tokens_by_request": cached,
        "output_token_hashes": [token_hash(seq.completion_token_ids) for seq in sequences],
        "output_tokens": [seq.num_completion_tokens for seq in sequences],
        "released_state": {"running": 0, "waiting": 0, "used_blocks": 0, "nonzero_refcounts": 0}}


def compare_reports(reports, expected_ids):
    """Fail closed on a missing case, changed contract, or any non-identical logit."""
    names = {"eager_no_prefix", "graph_no_prefix", "eager_prefix"}
    if set(reports) != names:
        raise ValueError("All three independent variant reports are required")
    baseline = reports["eager_no_prefix"]
    rows = []
    switches = {"eager_no_prefix": (True, False), "graph_no_prefix": (False, False), "eager_prefix": (True, True)}
    for name, report in reports.items():
        if report.get("status") != "pass" or report.get("evidence_class") != "functional_gpu_policy_validation":
            raise ValueError("Only completed numerical validation reports are accepted")
        if report["provenance"] != baseline["provenance"]:
            raise ValueError("Independent-process provenance differs")
        if report.get("variant") != name or (report["engine"].get("enforce_eager"), report["engine"].get("prefix_cache")) != switches[name]:
            raise ValueError("Variant flags do not match the independent process label")
        if report["engine"].get("dtype") != "bfloat16":
            raise ValueError("Validation must preserve BF16")
        common = lambda r: {k: v for k, v in r["engine"].items() if k not in ("enforce_eager", "prefix_cache")}
        if common(report) != common(baseline):
            raise ValueError("Physical pool or common engine budget changed")
        if set(report["cases"]) != set(expected_ids):
            raise ValueError("Validation case coverage is incomplete")
        for case in report["cases"].values():
            for phase in ("cold", "warm"):
                data = case["phases"][phase]
                hit = case["expected_warm_cached_tokens_per_request"] if name == "eager_prefix" and phase == "warm" else 0
                if data["cached_tokens_by_request"] != [hit] * case["batch_size"]:
                    raise ValueError("Prefix hit contract changed")
                if data["output_tokens"] != [64] * case["batch_size"] or any(data["released_state"].values()):
                    raise ValueError("Output budget or released state is invalid")
                if len(data["logits"]) != 64:
                    raise ValueError("Every one of the 64 full-vocabulary steps is required")
                for index, logit in enumerate(data["logits"]):
                    if (logit["step"] != index or logit["is_prefill"] != (index == 0)
                            or logit["dtype"] != "bfloat16" or logit["shape"][0] != case["batch_size"]
                            or len(logit["sha256"]) != 64):
                        raise ValueError("Invalid logit-step metadata")
    for case_id in expected_ids:
        reference = baseline["cases"][case_id]
        for name in ("graph_no_prefix", "eager_prefix"):
            candidate = reports[name]["cases"][case_id]
            if candidate["workload_sha256"] != reference["workload_sha256"]:
                raise ValueError("Teacher-forced input streams changed")
            for phase in ("cold", "warm"):
                expected = reference["phases"][phase]["logits"]
                actual = candidate["phases"][phase]["logits"]
                if len(expected) != 64 or len(actual) != 64:
                    raise ValueError("Every one of the 64 full-vocabulary steps is required")
                mismatches = [i for i, (a, b) in enumerate(zip(expected, actual))
                    if a != b]
                rows.append({"case_id": case_id, "variant": name, "phase": phase,
                    "compared_steps": 64, "exact": not mismatches, "mismatched_steps": mismatches})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--variant", choices=("eager_no_prefix", "graph_no_prefix", "eager_prefix"))
    args = parser.parse_args()
    validation, pilot, config, h1, cases = load_corrected_inputs(args.config)
    configure_compiler(validation)
    directory = ROOT / validation["results_dir"]
    if args.variant is None:
        reports = {name: json.loads((directory / f"{name}.json").read_text()) for name in validation["variants"]}
        rows = compare_reports(reports, [case["case_id"] for case in cases])
        passed = all(row["exact"] for row in rows)
        report = {"schema_version": 1, "stage": "H2", "status": "pass" if passed else "fail",
            "evidence_class": "corrected_policy_numerical_gate", "created_at_utc": now(),
            "policy_id": validation["policy_id"], "policy": validation["policy"],
            "provenance": reports["eager_no_prefix"]["provenance"],
            "common_engine": pilot["engine"], "allowed_variants": validation["variants"],
            "validated_case_ids": [case["case_id"] for case in cases],
            "comparisons": rows, "compared_logit_tensors": sum(r["compared_steps"] for r in rows),
            "gpu_execution_allowed": passed, "historical_native_gate_unchanged": True,
            "limitations": "Gate applies only to the explicit corrected policy and validation cases, not historical native performance or model quality."}
        write_result(report, directory / "gate.json")
        if not passed:
            raise SystemExit("Corrected policy has non-identical logits; see gate.json")
        print(f"Corrected policy: {report['compared_logit_tensors']} full-vocabulary tensor comparisons exact")
        return
    path = directory / f"{args.variant}.json"
    if path.exists():
        raise SystemExit("Select a fresh results_dir rather than overwrite numerical evidence")
    report = {"schema_version": 1, "stage": "H2", "status": "running", "variant": args.variant,
        "evidence_class": "functional_gpu_policy_validation", "created_at_utc": now(), "cases": {}}
    llm = None
    try:
        import torch
        torch.manual_seed(config["seed"])
        report["provenance"] = corrected_provenance(args.config, validation, pilot, config, h1)
        switches = config["variants"][args.variant]
        report["engine"] = {**pilot["engine"], **switches, "dtype": "bfloat16"}
        llm, report["initialization"] = create_corrected_engine(h1, pilot, switches, validation)
        if llm.model_runner.kv_cache.size(2) != pilot["engine"]["num_kvcache_blocks"]:
            raise AssertionError("Physical GPU pool changed")
        for case in cases:
            reset_cache(llm)
            llm.model_runner.kv_cache.fill_(float("nan"))
            phases = {phase: verify_wave(llm, case, phase, config) for phase in ("cold", "warm")}
            report["cases"][case["case_id"]] = {"workload_sha256": case["workload_sha256"],
                "prompt_stream_sha256": case["prompt_stream_sha256"],
                **{k: case[k] for k in ("batch_size", "prompt_length", "output_length", "prefix_mode", "expected_warm_cached_tokens_per_request")},
                "phases": phases}
            write_result(report, path)
            print(args.variant, case["case_id"], "64 cold + 64 warm steps passed", flush=True)
        report.update(status="pass", compile_counters=compile_counts(),
            compiler_settings={"recompile_limit": torch._dynamo.config.recompile_limit,
                "fail_on_recompile_limit_hit": torch._dynamo.config.fail_on_recompile_limit_hit,
                "suppress_errors": torch._dynamo.config.suppress_errors},
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
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
            write_result(report, path)


if __name__ == "__main__":
    main()

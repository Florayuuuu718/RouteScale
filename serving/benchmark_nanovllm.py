"""One isolated H2 preflight or formal pilot process, never a training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_runtime import (
    DEFAULT_PILOT, aggregate_waves, assert_released, check_capacity, close_engine,
    compile_counts, create_engine, execute_wave, load_inputs, now, provenance, reset_cache,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema_version": 1, "stage": "H2", "status": "running",
        "evidence_class": "gpu_preflight" if args.preflight else "formal_gpu",
        "case_id": args.case_id, "variant": args.variant, "run_id": args.run_id,
        "created_at_utc": now()}
    phase, llm = "input_validation", None
    try:
        pilot, config, h1, cases = load_inputs(args.config)
        case, switches = cases[args.case_id], config["variants"][args.variant]
        if args.variant not in pilot.get("variants", config["variants"]):
            raise ValueError("Variant is outside this pilot's explicit scope")
        if args.case_id not in config["pilot_case_ids"]:
            raise ValueError("The full matrix is not authorized by the pilot configuration")
        if args.case_id not in pilot.get("pilot_case_ids", config["pilot_case_ids"]):
            raise ValueError("Case is excluded from this correctness-qualified pilot")
        required = check_capacity(pilot["engine"], case)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA device for this pilot")
        torch.manual_seed(config["seed"])
        report["provenance"] = provenance(args.config, pilot, config, h1)
        report["engine"] = {**pilot["engine"], **switches, "dtype": config["dtype"]}
        report["workload"] = {"sha256": case["workload_sha256"],
            **{k: case[k] for k in ("batch_size", "prompt_length", "output_length", "prefix_mode")},
            "measured_pairs": config["measured_pairs_per_run"], "sampling": config["sampling"],
            "seed": config["seed"], "cache_reset_policy": config["cache_reset_policy"]}
        report["timing_policy"] = pilot["timing_policy"]
        preflight_path = ROOT / pilot["results_dir"] / "preflight.json"
        if not args.preflight:
            gate = json.loads(preflight_path.read_text())
            if (gate.get("status") != "pass" or not gate.get("gpu_execution_allowed")
                    or gate["provenance"] != report["provenance"]
                    or gate["common_engine"] != pilot["engine"]
                    or args.variant not in gate["allowed_variants"]
                    or (pilot.get("scope") == "cuda_graph_only" and args.case_id not in gate["validated_case_ids"])):
                raise ValueError("GPU preflight is missing, failed, or stale")
            report["preflight_sha256"] = digest(preflight_path)
        phase = "engine_initialization"
        llm, report["initialization"] = create_engine(h1, pilot, switches)
        actual = llm.model_runner.kv_cache.size(2)
        if actual != pilot["engine"]["num_kvcache_blocks"]:
            raise AssertionError("Physical GPU KV pool differs from configuration")
        report["engine"]["num_kvcache_blocks"] = actual
        report["capacity"] = {"cold_decode_required_blocks": required,
            "actual_physical_blocks": actual,
            "kv_tensor_bytes": llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()}

        if args.preflight:
            phase = "teacher_forced_numerical_probe"
            reset_cache(llm)
            logits = {}
            probe_waves = {}
            for name in ("cold", "warm"):
                probe_waves[name], logits[name] = execute_wave(llm, case, name, config,
                    forced_tokens=pilot["correctness_output_tokens"])
            path = ROOT / "runs/validation/h2_nanovllm" / Path(pilot["results_dir"]).name / f"{args.case_id}_{args.variant}_logits.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(logits, path)
            report["numerical_probe"] = {"path": str(path.relative_to(ROOT)),
                "sha256": digest(path), "forced_output_tokens": pilot["correctness_output_tokens"],
                "waves": probe_waves}

        phase = "excluded_workload_warmup"
        started = time.perf_counter()
        for _ in range(pilot["warmup_pairs"]):
            reset_cache(llm)
            for name in ("cold", "warm"):
                execute_wave(llm, case, name, config)
        torch.cuda.synchronize()
        report["workload_warmup_seconds"] = time.perf_counter() - started
        report["warmup_pairs"] = pilot["warmup_pairs"]

        phase = "measurements"
        before = compile_counts()
        waves = {"cold": [], "warm": []}
        # A preflight pair is functional evidence, not a formal benchmark result.
        pairs = 1 if args.preflight else config["measured_pairs_per_run"]
        for index in range(pairs):
            reset_cache(llm)
            torch.manual_seed(config["seed"] + index)
            for name in waves:
                row, _ = execute_wave(llm, case, name, config)
                waves[name].append(row)
        after = compile_counts()
        report["compile_counters"] = {"before_measurement": before, "after_measurement": after}
        if before != after:
            raise RuntimeError("New torch.compile work occurred inside the measurement window")
        report["waves"] = waves
        report["phases"] = {name: aggregate_waves(rows) for name, rows in waves.items()}
        assert_released(llm)
        report["released_state"] = {"running": 0, "waiting": 0, "used_blocks": 0, "nonzero_refcounts": 0}
        free, _ = torch.cuda.mem_get_info()
        report["free_bytes_after_run"] = free
        if free < pilot["minimum_free_after_kv_bytes"]:
            raise RuntimeError("GPU headroom dropped below the pilot safety margin")
        if not args.preflight:
            from scripts.summarize_h2_nanovllm import validate_run
            validate_run({**report, "status": "pass"})
        report["status"] = "pass"
    except BaseException:
        report.update(status="fail", failure_phase=phase, traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                close_engine(llm)
        except BaseException:
            report.update(status="fail", failure_phase="engine_cleanup", traceback=traceback.format_exc())
            raise
        finally:
            report["finished_at_utc"] = now()
            write_result(report, args.result)
    print(f"H2 {args.variant}/{args.case_id}/{args.run_id}: pass -> {args.result}", flush=True)


if __name__ == "__main__":
    main()

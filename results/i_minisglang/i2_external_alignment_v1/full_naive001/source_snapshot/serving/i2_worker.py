"""One fresh GPU process for reference, correctness, or controlled I2 timing."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs/vendor/mini-sglang-i2/python"))
from serving.i2_workloads import CONFIG, digest, token_hash, validate_workload


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def telemetry():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,utilization.gpu,driver_version",
                                   "--format=csv,noheader"], text=True).strip()


def error_metrics(actual, expected, policy):
    import torch
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("Every output step and vocabulary entry must match")
    results = []
    for index, (target, reference) in enumerate(zip(actual, expected, strict=True)):
        target, reference = target.float(), reference.float()
        finite = bool(torch.isfinite(target).all() and torch.isfinite(reference).all())
        if not finite:
            results.append(dict(step=index, finite=False, max_abs=None, relative_rmse=None,
                tolerance_violations=None, top1_equal=False, passed=False))
            continue
        diff = (target - reference).abs()
        relative_rmse = float((target - reference).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12))
        violations = int((diff > policy["atol"] + policy["rtol"] * reference.abs()).sum())
        results.append(dict(step=index, finite=finite, max_abs=float(diff.max()), relative_rmse=relative_rmse,
            tolerance_violations=violations, top1_equal=int(target.argmax()) == int(reference.argmax()),
            passed=finite and violations == 0 and relative_rmse <= policy["max_relative_rmse"]))
    return results


def reference(config, workload, validation):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(ROOT / config["model_path"], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").float().cuda().eval()
    tensors = {}
    for case in workload["cases"]:
        for phase in ("cold", "warm"):
            for request in case["waves"][phase]:
                inputs = torch.tensor([request["input_ids"]], device="cuda")
                past, values = None, []
                for forced in request["continuation_ids"]:
                    out = model(input_ids=inputs, past_key_values=past, use_cache=True, logits_to_keep=1)
                    values.append(out.logits[0, -1].detach().cpu().contiguous())
                    past = out.past_key_values
                    inputs = torch.tensor([[forced]], device="cuda")
                tensors[request["request_id"]] = torch.stack(values)
                del past, out
    path = validation / "reference.pt"
    torch.save(tensors, path)
    return dict(vectors=sum(v.shape[0] for v in tensors.values()), tensor_file=str(path.relative_to(ROOT)),
                tensor_sha256=digest(path), policy=config["numerics"])


def replay(config, workload, variant, mode, validation, raw, run_id):
    import torch
    from serving.i2_runtime import ReplayScheduler, assert_idle, flush_cache
    scheduler = None
    data = dict(cases=[], errors={}, capture=None)
    try:
        started = time.perf_counter()
        scheduler = ReplayScheduler(config, variant, capture_logits=mode == "numeric")
        data.update(initialization_s=time.perf_counter() - started,
            capture=dict(buckets=scheduler.engine.graph_runner.graph_bs_list,
                         max_bs=scheduler.engine.graph_runner.max_graph_bs),
            engine=dict(dtype=str(scheduler.engine.dtype), actual_backend=type(scheduler.engine.attn_backend).__name__,
                        num_pages=scheduler.engine.num_pages, max_seq_len=scheduler.engine.max_seq_len,
                        pool_shape=list(scheduler.engine.kv_cache._kv_buffer.shape)))
        reference_tensors = torch.load(validation / "reference.pt", mmap=True, weights_only=True) if mode == "numeric" else None
        repeats = 1 if mode == "numeric" else config["warmup_repeats"] + config["measure_repeats"]
        numeric_dir = validation / variant
        numeric_dir.mkdir(exist_ok=True)
        trace_path = raw / "batches.jsonl"
        cases = list(workload["cases"])
        random.Random(config["seed"] + run_id).shuffle(cases)
        for repeat in range(repeats):
            for case in cases:
                flush_cache(scheduler, poison=mode == "numeric")
                cache_start = assert_idle(scheduler)
                for phase in ("cold", "warm"):
                    row, tensors, batches = scheduler.wave(case["waves"][phase])
                    row.update(case_id=case["case_id"], phase=phase, repeat=repeat,
                        measured=mode == "benchmark" and repeat >= config["warmup_repeats"],
                        cache_empty_at_cold_start=cache_start["prefix_total"] == 0,
                        telemetry_after=telemetry())
                    expected_hit = config["shared_prefix_tokens"] if phase == "warm" and case["prefix_mode"] == "shared" and config["variants"][variant]["cache"] == "radix" else 0
                    if phase == "warm" and [r["cached_tokens"] for r in row["requests"]] != [expected_hit] * case["batch_size"]:
                        raise AssertionError("Warm prefix hit contract changed")
                    if phase == "cold" and config["variants"][variant]["cache"] == "naive" and any(r["cached_tokens"] for r in row["requests"]):
                        raise AssertionError("Naive cache unexpectedly reused prefixes")
                    with trace_path.open("a") as stream:
                        stream.write(json.dumps(dict(case_id=case["case_id"], repeat=repeat, phase=phase, batches=batches)) + "\n")
                    if mode == "numeric":
                        path = numeric_dir / f"{case['case_id']}_{phase}.pt"
                        torch.save(tensors, path)
                        row.update(tensor_file=str(path.relative_to(ROOT)), tensor_sha256=digest(path))
                        for rid, tensor in tensors.items():
                            data["errors"][rid] = error_metrics(tensor, reference_tensors[rid], config["numerics"])
                    data["cases"].append(row)
                print(variant, mode, repeat, case["case_id"], "cold/warm completed", flush=True)
        data["trace_sha256"] = digest(trace_path)
        data["released"] = assert_idle(scheduler)
        data["status"] = "pass" if mode == "benchmark" or all(r["passed"] for errors in data["errors"].values() for r in errors) else "fail"
        return data
    finally:
        if scheduler is not None:
            started = time.perf_counter()
            scheduler.shutdown()
            data["engine_shutdown_s"] = time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reference", "numeric", "benchmark"], required=True)
    parser.add_argument("--variant")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--run", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    workload_path = ROOT / config["workload_path"]
    workload = json.loads(workload_path.read_text())
    validate_workload(workload, config)
    from scripts.prepare_i2_minisglang import validate
    series = validate(ROOT / config["checkout"])
    label = "reference" if args.mode == "reference" else f"{args.mode}_{args.variant}_run{args.run}"
    directory = ROOT / config["results_dir"] / args.attempt
    output = directory / f"{label}.json"
    raw = ROOT / config["runtime_dir"] / args.attempt / label
    validation = ROOT / config["validation_dir"] / args.attempt
    if output.exists():
        raise ValueError("Preserve existing worker evidence")
    raw.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    record = dict(stage="I2", mode=args.mode, variant=args.variant, run_id=args.run, status="fail",
        config_sha256=digest(CONFIG), workload_sha256=digest(workload_path), patched_tree=series["tree"],
        runtime_dir=str(raw.relative_to(ROOT)), gpu_before=telemetry(),
        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "flashinfer-python", "sgl-kernel")},
        source_sha256={name: digest(ROOT / name) for name in ("serving/i2_worker.py", "serving/i2_runtime.py", "serving/i2_workloads.py")})
    try:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["numerics"]["allow_bf16_reduced_precision_reduction"]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record["precision"] = dict(bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                   tf32=torch.backends.cuda.matmul.allow_tf32)
        with torch.inference_mode():
            if args.mode == "reference":
                record["reference"] = reference(config, workload, validation)
                record["status"] = "pass"
            else:
                record["replay"] = replay(config, workload, args.variant, args.mode, validation, raw, args.run)
                record["status"] = record["replay"]["status"]
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["gpu_end_of_worker"] = telemetry()
    except BaseException:
        record["error"] = traceback.format_exc()
        print(record["error"], flush=True)
    finally:
        write(output, record)
    print(json.dumps(dict(label=label, status=record["status"])), flush=True)
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

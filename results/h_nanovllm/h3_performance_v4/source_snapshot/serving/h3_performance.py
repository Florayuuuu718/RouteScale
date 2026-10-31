"""New, isolated H3 performance scope: full-BF16 fixed-token controls."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback

from serving.backends.nanovllm import ROOT, digest, load_config, validate_environment, write_result

DEFAULT_CONFIG = ROOT / "config/serving/h3_performance_v4.json"
VARIANTS = ("native_kv", "paged_eager", "paged_graph")
SCOPES = ("model_execution", "replay_e2e")
REFERENCE_POLICIES = ("cohort_full_recompute", "single_request_full_recompute", "bf16_weights_fp32_math")
SOURCES = (
    "model.py", "moe.py", "distributed_common.py", "serving/artifact.py",
    "serving/h3_model.py", "serving/backends/nanovllm.py",
    "serving/backends/routescale_nanovllm.py", "serving/h2_numerical_policy.py",
    "serving/h2_prefix_probe.py", "serving/h2_runtime.py", "serving/native/benchmark.py",
    "serving/h3_performance.py", "scripts/run_h3_performance.py",
    "scripts/summarize_h3_performance.py", "scripts/audit_h3_performance.py",
    "docs/h3_performance_protocol.md",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config):
    if (config["schema_version"] != 1 or config["stage"] != "H3"
            or config["dtype"] != "bfloat16" or config["prefix_cache"] is not False
            or tuple(config["variants"]) != VARIANTS
            or config["token_policy"] != "argmax_diagnostic_then_fixed_validation_token_replay"):
        raise ValueError("Unsupported H3 precision, token policy or controlled variants")
    if config.get("reference_policy", "cohort_full_recompute") not in REFERENCE_POLICIES:
        raise ValueError("Unknown full-recompute reference policy")
    if type(config.get("allow_bf16_reduced_precision_reduction", True)) is not bool:
        raise ValueError("The BF16 reduction permission must be a boolean")
    additional = config.get("additional_reference_policies", [])
    if (not set(additional) <= set(REFERENCE_POLICIES) or len(set(additional)) != len(additional)
            or config.get("reference_policy", "cohort_full_recompute") in additional):
        raise ValueError("Invalid additional references")
    engine = config["engine"]
    if (engine["tensor_parallel_size"] != 1 or engine["max_model_len"] != 512
            or engine["kvcache_block_size"] != 256 or engine["num_kvcache_blocks"] != 16):
        raise ValueError("H3 requires TP=1, a 512-position window and the fixed 64 MiB pool")
    names = [case["case_id"] for case in config["cases"]]
    if len(names) != len(set(names)) or not set(config["pilot_case_ids"]) <= set(names):
        raise ValueError("Duplicate cases or unknown pilot case")
    for case in config["cases"]:
        b, p, o = (case[k] for k in ("batch_size", "prompt_length", "output_length"))
        if any(type(v) is not int or v <= 0 for v in (b, p, o)) or o < 2 or p + o - 1 > 512:
            raise ValueError("Invalid request budget or cached-position window overflow")
        if engine["max_num_seqs"] < b or engine["max_num_batched_tokens"] < b * p:
            raise ValueError("Engine cannot admit the complete Prefill cohort")
        if b * math.ceil((p + o - 1) / 256) > engine["num_kvcache_blocks"]:
            raise ValueError("KV pool cannot fund future page growth")
    if (config["independent_runs"] < 3 or config["warmup_repeats"] < 3
            or config["measure_repeats"] < 5):
        raise ValueError("Insufficient independent runs or steady-state repeats")
    return config


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def new_output(path):
    if path.exists():
        raise ValueError(f"Preserve existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare(path):
    from serving.native.benchmark import build_prompt_batch, build_decode_batch
    config = read_config(path)
    output = ROOT / config["results_dir"] / "preparation.json"
    workload = ROOT / config["workload_path"]
    new_output(output)
    new_output(workload)
    cases = []
    for index, shape in enumerate(config["cases"]):
        prompts, _ = build_prompt_batch(ROOT / config["data_path"], batch_size=shape["batch_size"],
            prompt_length=shape["prompt_length"], prefix_mode="none", seed=config["seed"] + index * 2)
        continuation, _ = build_decode_batch(ROOT / config["data_path"], batch_size=shape["batch_size"],
            output_length=shape["output_length"], seed=config["seed"] + index * 2 + 1)
        case = dict(**shape, prompt_ids=prompts.tolist(), continuation_ids=continuation.tolist())
        cases.append(dict(**case, workload_sha256=value_hash(case)))
    write_result(dict(schema_version=1, stage="H3", cases=cases), workload)
    write_result(dict(schema_version=1, stage="H3", status="pass", created_at_utc=now(),
        config_sha256=digest(path), builder_sha256=digest(Path(__file__)),
        data_sha256=digest(ROOT / config["data_path"]), workload_sha256=digest(workload),
        tokenizer_sha256=digest(ROOT / config["tokenizer_path"])), output)
    print("H3 performance workload: pass", flush=True)


def load_inputs(path):
    import torch
    config = read_config(path)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config.get(
        "allow_bf16_reduced_precision_reduction", True)
    pin = load_config(ROOT / config["h1_config"])
    validate_environment(pin)
    preparation_path = ROOT / config["results_dir"] / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    if (preparation["status"] != "pass" or preparation["config_sha256"] != digest(path)
            or preparation["builder_sha256"] != digest(Path(__file__))
            or preparation["workload_sha256"] != digest(ROOT / config["workload_path"])
            or preparation["data_sha256"] != digest(ROOT / config["data_path"])
            or preparation["tokenizer_sha256"] != digest(ROOT / config["tokenizer_path"])):
        raise ValueError("Prepared workload/configuration/source changed")
    cases = json.loads((ROOT / config["workload_path"]).read_text())["cases"]
    if [{k: c[k] for k in config["cases"][0]} for c in cases] != config["cases"]:
        raise ValueError("Prepared shapes changed")
    for case in cases:
        if value_hash({k: v for k, v in case.items() if k != "workload_sha256"}) != case["workload_sha256"]:
            raise ValueError("Workload token arrays changed")
    manifest_path = ROOT / config["artifact_manifest"]
    manifest = json.loads(manifest_path.read_text())
    artifact = ROOT / manifest["artifact_path"]
    if digest(artifact) != config["artifact_sha256"] or digest(artifact) != manifest["artifact_sha256"]:
        raise ValueError("Original inference artifact changed")
    upstream = ROOT / pin["upstream"]["path"]
    provenance = dict(config_sha256=digest(path), artifact_sha256=digest(artifact),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        reference_policy=config.get("reference_policy", "cohort_full_recompute"),
        protocol_sha256=digest(ROOT / config.get("protocol", "docs/h3_performance_protocol.md")),
        historical_baseline_sha256=digest(ROOT / config["results_dir"] / "historical_baseline.json"),
        manifest_sha256=digest(manifest_path), preparation_sha256=digest(preparation_path),
        workload_sha256=preparation["workload_sha256"], data_sha256=preparation["data_sha256"],
        tokenizer_sha256=preparation["tokenizer_sha256"], upstream_commit=pin["upstream"]["commit"],
        upstream_source_sha256={str(p.relative_to(upstream)): digest(p) for p in sorted(upstream.rglob("*.py"))},
        source_sha256={name: digest(ROOT / name) for name in SOURCES},
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
            packages=json.loads(subprocess.check_output(["runs/envs/nano-vllm/bin/python", "-m", "pip", "list", "--format=json"], text=True))))
    return config, artifact, cases, provenance


def load_native(artifact):
    import torch
    from serving.artifact import load_inference_artifact
    native, _ = load_inference_artifact(artifact, device="cpu", dtype=torch.float32)
    return native.to(device="cuda", dtype=torch.bfloat16).eval()


def full_reference_logits(model, inputs, policy):
    """Explicit reference batching; the caller controls compute dtype/backend."""
    import torch
    if policy in ("single_request_full_recompute", "bf16_weights_fp32_math"):
        return torch.cat([model(row.unsqueeze(0))[0][:, -1] for row in inputs], dim=0)
    if policy == "cohort_full_recompute":
        return model(inputs)[0][:, -1]
    raise ValueError("Unknown full-recompute reference policy")


class NativePath:
    def __init__(self, artifact, case):
        self.model = load_native(artifact)
        self.cache = self.model.allocate_kv_cache(case["batch_size"])
        self.kv_bytes = self.cache.allocated_bytes
        self.initialization = dict(graph_capture_seconds=0.0)

    def start(self, case):
        self.case = case
        self.cache.reset()
        self.generated = [p.copy() for p in case["prompt_ids"]]
        self.step = 0

    def poison(self):
        for tensor in (*self.cache.keys, *self.cache.values):
            tensor.fill_(float("nan"))

    def prepare(self):
        import torch
        ids = self.generated if self.step == 0 else [[row[-1]] for row in self.generated]
        return torch.tensor(ids, dtype=torch.long, device="cuda")

    def compute(self, ids):
        return self.model.forward_with_kv_cache(ids, self.cache)[:, -1]

    def usage(self):
        return dict(cached_positions=self.cache.length, used_blocks=0, prefix_tokens=0)

    def advance(self, tokens):
        for row, token in zip(self.generated, tokens):
            row.append(token)
        self.step += 1

    def finish(self):
        if self.cache.length != self.case["prompt_length"] + self.case["output_length"] - 1:
            raise ValueError("Native cached length changed")
        outputs = [row[self.case["prompt_length"]:] for row in self.generated]
        self.generated = []
        return outputs

    def close(self):
        del self.model, self.cache


class PagedPath:
    def __init__(self, artifact, case, config, variant):
        import torch
        from nanovllm.engine.model_runner import ModelRunner
        from serving.backends.routescale_nanovllm import create_engine
        from serving.h3_model import OfflineGPT2Tokenizer
        self.initialization = dict(graph_capture_seconds=0.0)
        original = ModelRunner.capture_cudagraph

        def capture(runner):
            started = time.perf_counter()
            original(runner)
            torch.cuda.synchronize()
            self.initialization["graph_capture_seconds"] = time.perf_counter() - started

        ModelRunner.capture_cudagraph = capture
        try:
            tokenizer = OfflineGPT2Tokenizer(json.loads((ROOT / config["tokenizer_path"]).read_text()))
            self.engine = create_engine(config["engine"], artifact, tokenizer,
                enforce_eager=variant != "paged_graph", prefix_cache=False)
        finally:
            ModelRunner.capture_cudagraph = original
        self.model = self.engine.model_runner.model.gpt
        self.kv_bytes = self.engine.model_runner.kv_cache.numel() * 2

        def reject_preemption(_):
            raise ValueError("Unexpected preemption in a fully funded cohort")

        self.engine.scheduler.preempt = reject_preemption

    def start(self, case):
        from nanovllm import SamplingParams
        self.case, self.step = case, 0
        self.engine.reset_cache()
        for prompt in case["prompt_ids"]:
            self.engine.add_request(prompt, SamplingParams(temperature=.6,
                max_tokens=case["output_length"], ignore_eos=True))
        self.sequences = list(self.engine.scheduler.waiting)

    def poison(self):
        self.engine.model_runner.kv_cache.fill_(float("nan"))

    def prepare(self):
        seqs, prefill = self.engine.scheduler.schedule()
        if seqs != self.sequences or prefill != (self.step == 0):
            raise ValueError("Incomplete cohort, split Prefill or reordered requests")
        if prefill and any(seq.num_cached_tokens for seq in seqs):
            raise ValueError("Prefix hits must be zero")
        self.active, self.prefill = seqs, prefill
        runner = self.engine.model_runner
        return runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)

    def compute(self, prepared):
        return self.engine.model_runner.run_model(*prepared, self.prefill)

    def usage(self):
        return dict(cached_positions=max(s.num_cached_tokens + s.num_scheduled_tokens for s in self.active),
            used_blocks=len(self.engine.scheduler.block_manager.used_block_ids), prefix_tokens=0)

    def advance(self, tokens):
        from nanovllm.utils.context import reset_context
        reset_context()
        self.engine.scheduler.postprocess(self.active, tokens, self.prefill)
        self.step += 1

    def finish(self):
        from serving.h2_runtime import assert_released
        assert_released(self.engine)
        outputs = [seq.completion_token_ids for seq in self.sequences]
        self.active, self.sequences = [], []
        return outputs

    def close(self):
        from serving.backends.routescale_nanovllm import close_engine
        del self.model
        close_engine(self.engine)
        del self.engine


def create_path(artifact, case, config, variant):
    import torch
    from distributed_common import state_dict_sha256
    started = time.perf_counter()
    path = NativePath(artifact, case) if variant == "native_kv" else PagedPath(artifact, case, config, variant)
    torch.cuda.synchronize()
    path.initialization["total_seconds"] = time.perf_counter() - started
    state = path.model.state_dict()
    if any(t.dtype != torch.bfloat16 for t in state.values()):
        raise ValueError("Every model weight must be BF16")
    if path.model.lm_head.weight is not path.model.transformer.wte.weight:
        raise ValueError("Tied weights changed")
    path.weight_sha256 = state_dict_sha256(state)
    return path


def run_wave(path, case, scope, *, collect_logits=False, poison=False):
    import torch
    if scope not in SCOPES:
        raise ValueError("Unknown timing scope")
    torch.cuda.synchronize()
    started = time.perf_counter()
    path.start(case)
    if poison:
        path.poison()
    times, events, rows, predictions, usage = [], [], [], [], []
    with torch.inference_mode():
        for step in range(case["output_length"]):
            prepared = path.prepare()
            if scope == "model_execution":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            logits = path.compute(prepared)
            if scope == "model_execution":
                end.record()
                events.append((begin, end))
            if collect_logits:
                rows.append(logits.detach().cpu().contiguous())
            if scope == "replay_e2e":
                predictions.append(torch.argmax(logits, dim=-1).tolist())
            usage.append(path.usage())
            path.advance([row[step] for row in case["continuation_ids"]])
            if scope == "replay_e2e":
                times.append((time.perf_counter() - started) * 1000)
        outputs = path.finish()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    if outputs != case["continuation_ids"]:
        raise ValueError("Actual output tokens differ from the frozen replay")
    if scope == "model_execution":
        step_ms = [a.elapsed_time(b) for a, b in events]
        ttft, itl = step_ms[0], step_ms[1:]
        execution_ms = sum(step_ms)
    else:
        ttft, itl = times[0], [b - a for a, b in zip(times, times[1:])]
        execution_ms = None
    trace = dict(ttft_ms=ttft, itl_ms=itl, tpot_ms=sum(itl) / len(itl),
        wall_ms=wall_ms, model_execution_ms=execution_ms,
        output_tokens=case["batch_size"] * case["output_length"],
        output_sha256=value_hash(outputs), argmax_sha256=value_hash(predictions) if predictions else None,
        cached_positions=max(u["cached_positions"] for u in usage),
        max_used_blocks=max(u["used_blocks"] for u in usage), prefix_tokens=0,
        prefill_calls=1, decode_calls=case["output_length"] - 1, released=True)
    return trace, torch.stack(rows) if rows else None, usage


def tensor_metadata(tensor, path):
    import torch
    new_output(path)
    torch.save(tensor, path)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path), shape=list(tensor.shape), dtype=str(tensor.dtype))


def reference_name(config, policy):
    return "reference.json" if policy == config.get("reference_policy", "cohort_full_recompute") else f"reference_{policy}.json"


def reference_comparison_key(config, policy, variant, case_id):
    prefix = "" if policy == config.get("reference_policy", "cohort_full_recompute") else f"{policy}/"
    return f"{prefix}{variant}/{case_id}"


def reference(path, *, policy_override=None):
    import torch
    from distributed_common import state_dict_sha256
    config, artifact, cases, provenance = load_inputs(path)
    policy = policy_override or config.get("reference_policy", "cohort_full_recompute")
    if policy not in (config.get("reference_policy", "cohort_full_recompute"), *config.get("additional_reference_policies", [])):
        raise ValueError("Reference policy is not qualified by this configuration")
    output = ROOT / config["results_dir"] / reference_name(config, policy)
    new_output(output)
    model = load_native(artifact)
    report = dict(schema_version=1, stage="H3", status="running", created_at_utc=now(),
        provenance=provenance, weight_sha256=state_dict_sha256(model.state_dict()), cases={})
    fp32_reference = policy == "bf16_weights_fp32_math"
    if fp32_reference:
        # Promote the already-rounded BF16 parameters. Never substitute the
        # original FP32 checkpoint values for the BF16 inference weights.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model.float()
        math_context = sdpa_kernel(SDPBackend.MATH)
    else:
        math_context = nullcontext()
    report.update(reference_policy=policy, reference_dtype="float32" if fp32_reference else "bfloat16")
    try:
        with torch.inference_mode(), math_context:
            for case in cases:
                generated = torch.tensor(case["prompt_ids"], device="cuda")
                rows = []
                forced = torch.tensor(case["continuation_ids"], device="cuda")
                for step in range(case["output_length"]):
                    rows.append(full_reference_logits(model, generated, policy).cpu().contiguous())
                    generated = torch.cat((generated, forced[:, step:step + 1]), dim=1)
                tensor = torch.stack(rows)
                expected_dtype = torch.float32 if fp32_reference else torch.bfloat16
                if not bool(torch.isfinite(tensor).all()) or tensor.dtype != expected_dtype:
                    raise ValueError("Nonfinite or wrong-precision native reference")
                prefix = "reference" if output.name == "reference.json" else f"reference_{policy}"
                report["cases"][case["case_id"]] = tensor_metadata(tensor,
                    ROOT / config["validation_dir"] / f"{prefix}_{case['case_id']}.pt")
        report["status"] = "pass"
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        write_result(report, output)
    print(f"H3 {report['reference_dtype']} reference ({policy}): pass", flush=True)


def qualify(path, variant):
    import torch
    config, artifact, cases, provenance = load_inputs(path)
    output = ROOT / config["results_dir"] / "qualification" / f"{variant}.json"
    new_output(output)
    report = dict(schema_version=1, stage="H3", status="running", created_at_utc=now(),
        variant=variant, provenance=provenance, cases={})
    session = None
    try:
        # One new model per case keeps the native B-dependent allocation explicit.
        for case in cases:
            session = create_path(artifact, case, config, variant)
            trace, tensor, usage = run_wave(session, case, "model_execution", collect_logits=True, poison=True)
            metadata = tensor_metadata(tensor, ROOT / config["validation_dir"] / f"{variant}_{case['case_id']}.pt")
            report["cases"][case["case_id"]] = dict(**metadata, trace=trace, usage=usage,
                kv_bytes=session.kv_bytes, weight_sha256=session.weight_sha256)
            session.close()
            session = None
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "pass"
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        if session is not None:
            session.close()
        write_result(report, output)
    print(f"H3 qualification tensors: {variant} pass", flush=True)


def compare_tensors(actual, expected, tolerance, *, exact=False, fp32_reference=False):
    import torch
    expected_dtype = torch.float32 if fp32_reference else torch.bfloat16
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16 or expected.dtype != expected_dtype:
        raise ValueError("Logit shape or dtype mismatch")
    a, b = actual.float(), expected.float()
    diff = a - b
    relative = diff.square().mean(-1).sqrt() / b.square().mean(-1).sqrt().clamp_min(1e-12)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    bitwise = torch.equal(actual, expected)
    passed = finite and (bitwise if exact else bool(torch.allclose(a, b,
        atol=tolerance["bf16_atol"], rtol=tolerance["bf16_rtol"]))
        and float(relative.max()) <= tolerance["max_relative_rmse"])
    return dict(passed=passed, vectors=actual.shape[0] * actual.shape[1], exact=bitwise,
        max_absolute_error=float(diff.abs().max()), max_relative_rmse=float(relative.max()))


def compare(path):
    import torch
    config, _, cases, provenance = load_inputs(path)
    directory = ROOT / config["results_dir"]
    output = directory / "gate.json"
    new_output(output)
    reference_path = directory / "reference.json"
    ref = json.loads(reference_path.read_text())
    policies = [config.get("reference_policy", "cohort_full_recompute"), *config.get("additional_reference_policies", [])]
    references = {policy: json.loads((directory / reference_name(config, policy)).read_text()) for policy in policies}
    reports = {v: json.loads((directory / "qualification" / f"{v}.json").read_text()) for v in VARIANTS}
    gate = dict(schema_version=1, stage="H3", status="fail", created_at_utc=now(), provenance=provenance,
        performance_execution_allowed=False, comparisons={}, report_sha256={"reference.json": digest(reference_path)})

    def read_tensor(metadata, case):
        target = ROOT / metadata["path"]
        if digest(target) != metadata["sha256"]:
            raise ValueError("Saved logits changed")
        tensor = torch.load(target, weights_only=True, mmap=True, map_location="cpu")
        if list(tensor.shape) != [case["output_length"], case["batch_size"], 50257]:
            raise ValueError("Incomplete full-step full-vocabulary coverage")
        return tensor

    try:
        for report in [*references.values(), *reports.values()]:
            if report["status"] != "pass" or report["provenance"] != provenance or set(report["cases"]) != {c["case_id"] for c in cases}:
                raise ValueError("Incomplete or stale qualification")
        for policy, report in references.items():
            if report["reference_policy"] != policy or report["weight_sha256"] != ref["weight_sha256"]:
                raise ValueError("Reference policy or BF16 parameter values changed")
            gate["report_sha256"][reference_name(config, policy)] = digest(directory / reference_name(config, policy))
        for variant, report in reports.items():
            gate["report_sha256"][f"qualification/{variant}.json"] = digest(directory / "qualification" / f"{variant}.json")
            for case in cases:
                name = case["case_id"]
                metadata = report["cases"][name]
                if metadata["weight_sha256"] != ref["weight_sha256"]:
                    raise ValueError("BF16 parameter values differ")
                trace = metadata["trace"]
                if (not trace["released"] or trace["prefix_tokens"] != 0
                        or trace["cached_positions"] != case["prompt_length"] + case["output_length"] - 1):
                    raise ValueError("Cache position/prefix/reclamation gate failed")
                if variant != "native_kv" and trace["max_used_blocks"] != case["batch_size"] * math.ceil(trace["cached_positions"] / 256):
                    raise ValueError("Paged allocation did not cover the expected growth")
                actual = read_tensor(metadata, case)
                for policy, reference_report in references.items():
                    key = reference_comparison_key(config, policy, variant, name)
                    gate["comparisons"][key] = compare_tensors(actual,
                        read_tensor(reference_report["cases"][name], case), config["tolerances"],
                        fp32_reference=policy == "bf16_weights_fp32_math")
                if variant == "paged_graph":
                    gate["comparisons"][f"graph_exact/{name}"] = compare_tensors(actual,
                        read_tensor(reports["paged_eager"]["cases"][name], case), config["tolerances"], exact=True)
        if not all(c["passed"] for c in gate["comparisons"].values()):
            raise ValueError("New H3 numerical gate failed; no performance runs permitted")
        gate.update(status="pass", performance_execution_allowed=True,
            native_bf16_vectors=sum(c["vectors"] for k, c in gate["comparisons"].items() if k.split("/")[0] in VARIANTS),
            reference_comparison_vectors=sum(c["vectors"] for k, c in gate["comparisons"].items() if not k.startswith("graph_exact/")),
            eager_graph_exact_vectors=sum(c["vectors"] for k, c in gate["comparisons"].items() if k.startswith("graph_exact/")))
    except BaseException:
        gate["traceback"] = traceback.format_exc()
        raise
    finally:
        write_result(gate, output)
    print(f"H3 new performance gate: pass, {gate['native_bf16_vectors']} native vectors", flush=True)


def benchmark(path, variant, case_id, run_id, *, pilot=False):
    import torch
    config, artifact, cases, provenance = load_inputs(path)
    case = next(c for c in cases if c["case_id"] == case_id)
    directory = ROOT / config["results_dir"]
    gate = json.loads((directory / "gate.json").read_text())
    if gate["status"] != "pass" or not gate["performance_execution_allowed"] or gate["provenance"] != provenance:
        raise ValueError("Performance qualification missing or stale")
    if not pilot:
        pilot_gate = json.loads((directory / "pilot_gate.json").read_text())
        if pilot_gate["status"] != "pass" or pilot_gate["qualification_gate_sha256"] != digest(directory / "gate.json"):
            raise ValueError("Pilot must pass before formal timing")
    output = directory / ("pilot" if pilot else "raw") / case_id / variant / f"{run_id}.json"
    new_output(output)
    report = dict(schema_version=1, stage="H3", benchmark="routescale_performance", status="running",
        evidence_class="pilot_gpu" if pilot else "formal_gpu", created_at_utc=now(),
        variant=variant, case_id=case_id, run_id=run_id, provenance=provenance,
        workload_sha256=case["workload_sha256"], qualification_gate_sha256=digest(directory / "gate.json"),
        measurements={}, dtype="bfloat16", prefix_cache=False, token_policy=config["token_policy"])
    session = None
    try:
        session = create_path(artifact, case, config, variant)
        report.update(initialization=session.initialization, kv_bytes=session.kv_bytes,
            weight_sha256=session.weight_sha256)
        if report["weight_sha256"] != json.loads((directory / "reference.json").read_text())["weight_sha256"]:
            raise ValueError("Benchmark weights differ from qualification")
        warmup = config["pilot_warmup_repeats" if pilot else "warmup_repeats"]
        measured = config["pilot_measure_repeats" if pilot else "measure_repeats"]
        report.update(warmup_repeats=warmup, measure_repeats=measured)
        for scope in SCOPES:
            for _ in range(warmup):
                run_wave(session, case, scope)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            traces = [run_wave(session, case, scope)[0] for _ in range(measured)]
            if len({t["output_sha256"] for t in traces}) != 1:
                raise ValueError("Non-deterministic replay outputs")
            report["measurements"][scope] = dict(traces=traces,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report["free_bytes_after_run"] = torch.cuda.mem_get_info()[0]
        if report["free_bytes_after_run"] < config["minimum_free_bytes"]:
            raise ValueError("Insufficient steady-state memory headroom")
        report["status"] = "pass"
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        if session is not None:
            session.close()
        report["finished_at_utc"] = now()
        write_result(report, output)
    print(f"H3 {'pilot' if pilot else 'formal'}: {case_id}/{variant}/{run_id} pass", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "reference", "qualify", "compare", "pilot", "benchmark"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--case-id")
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--reference-policy", choices=REFERENCE_POLICIES)
    args = parser.parse_args()
    if args.command in ("pilot", "benchmark"):
        benchmark(args.config, args.variant, args.case_id, args.run_id, pilot=args.command == "pilot")
    elif args.command == "qualify":
        qualify(args.config, args.variant)
    elif args.command == "reference":
        reference(args.config, policy_override=args.reference_policy)
    else:
        globals()[args.command](args.config)


if __name__ == "__main__":
    main()

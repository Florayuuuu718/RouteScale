"""Arrival replay and admission-only controls; failed v1 sources are archived."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from serving.backends.nanovllm import ROOT, digest
from serving.h2_corrected_runtime import (
    close_corrected_engine, corrected_provenance, create_corrected_engine, load_corrected_inputs,
)
from serving.h2_linear_policy import install_prefill_linear_policy, validate_linear_policy
from serving.h2_runtime import assert_released, quantile, reset_cache
from serving.h2_workloads import token_hash

DEFAULT_SCHEDULING = ROOT / "config/serving/h2_scheduling.json"
SCHEDULING_SOURCES = (
    "serving/h2_scheduling.py", "serving/benchmark_h2_scheduling.py",
    "scripts/run_h2_scheduling.py", "scripts/summarize_h2_scheduling.py",
    "serving/h2_linear_policy.py",
)
POLICIES = ("fixed_cohort", "continuous")
SWITCHES = dict(enforce_eager=False, prefix_cache=False)


def admission_count(policy, active, limit, pending, elapsed_ms):
    if policy not in POLICIES or not 0 <= active <= limit:
        raise ValueError("Invalid admission policy or active request count")
    if policy == "fixed_cohort" and active:
        return 0
    due = sum(request["arrival_ms"] <= elapsed_ms for request in pending)
    return min(limit - active, due)


def build_cases(config, prompts):
    if (config["schema_version"] != 1 or config["logical_validation_step_ms"] != 10
            or config["max_active_requests"] != 8 or len(prompts) != 16
            or config["policies"] != list(POLICIES) or config["independent_runs"] != 3
            or config["measured_traces"] != 3 or config["warmup_traces"] != 1):
        raise ValueError("This pilot fixes 16 requests, eight slots, two policies and three runs/traces")
    expected = (("equal_burst", [32], 0),
        ("mixed_burst", [8, 8, 8, 8, 64, 64, 64, 64], 0),
        ("mixed_staggered", [8, 8, 8, 8, 64, 64, 64, 64], 160))
    if [(c["case_id"], c["output_pattern"], c["second_wave_arrival_ms"]) for c in config["cases"]] != list(expected):
        raise ValueError("Do not silently expand or replace the fixed scheduling scenarios")
    cases = {}
    for dimensions in config["cases"]:
        pattern = dimensions["output_pattern"]
        requests = []
        for index, prompt in enumerate(prompts):
            if len(prompt) != 128:
                raise ValueError("All scheduling prompts must contain exactly 128 tokens")
            requests.append(dict(request_id=f"r{index:02}", prompt_token_ids=prompt,
                output_tokens=pattern[index % len(pattern)],
                arrival_ms=dimensions["second_wave_arrival_ms"] if index >= 8 else 0))
        case = dict(case_id=dimensions["case_id"], requests=requests)
        case["workload_sha256"] = token_hash(case)
        cases[case["case_id"]] = case
    return cases


def load_scheduling_inputs(path=DEFAULT_SCHEDULING):
    config = json.loads(path.read_text())
    validate_linear_policy(config.get("linear_policy"))
    if config.get("linear_policy"):
        previous = ROOT / config["previous_failed_results"]
        if digest(previous / "preflight.json") != config["previous_failed_gate_sha256"]:
            raise ValueError("Previous failed scheduling gate changed")
        manifest_path = previous / "source_snapshot/manifest.json"
        if digest(manifest_path) != config["failed_source_manifest_sha256"]:
            raise ValueError("Failed-harness source snapshot manifest changed")
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"].values():
            if digest(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Original failed harness snapshot changed")
        if (ROOT / config["results_dir"]).resolve() == previous.resolve():
            raise ValueError("Corrected runs must not overwrite the original failure directory")
    validation_path = ROOT / config["base_validation"]
    validation, pilot, base, h1, _ = load_corrected_inputs(validation_path)
    base_gate_path = ROOT / validation["results_dir"] / "gate.json"
    if digest(base_gate_path) != config["base_gate_sha256"]:
        raise ValueError("Corrected-policy base gate changed")
    gate = json.loads(base_gate_path.read_text())
    if gate["status"] != "pass" or not gate["gpu_execution_allowed"]:
        raise ValueError("Corrected-policy base qualification failed")
    for field in ("source_sha256", "corrected_source_sha256"):
        for name, checksum in gate["provenance"][field].items():
            if digest(ROOT / name) != checksum:
                raise ValueError(f"Qualified source changed: {name}")
    fixture = json.loads((ROOT / base["workload_path"]).read_text())
    source_case = next(c for c in fixture["cases"] if c["case_id"] == config["prompt_case_id"])
    prompts = [r["prompt_token_ids"] for phase in ("cold", "warm") for r in source_case["waves"][phase]]
    cases = build_cases(config, prompts)
    engine = pilot["engine"]
    if (engine["max_num_seqs"] != 8 or engine["max_num_batched_tokens"] < 8 * 128
            or engine["max_model_len"] < 192 or engine["num_kvcache_blocks"] < 8):
        raise ValueError("Common engine cannot cover the entire eight-slot request budget")
    return config, validation, pilot, base, h1, cases, gate


def scheduling_provenance(path, inputs):
    config, validation, pilot, base, h1, _, gate = inputs
    current = corrected_provenance(ROOT / config["base_validation"], validation, pilot, base, h1)
    if current != gate["provenance"]:
        raise ValueError("Model, environment or corrected runtime differs from qualification")
    return {**current, "scheduling_config_sha256": digest(path),
        "prefill_linear_policy": config.get("linear_policy"),
        "scheduling_source_sha256": {name: digest(ROOT / name) for name in SCHEDULING_SOURCES}}


def create_scheduling_engine(h1, pilot, validation, config):
    llm, timings = create_corrected_engine(h1, pilot, SWITCHES, validation)
    try:
        if config.get("linear_policy"):
            from nanovllm.layers.linear import LinearBase
            from nanovllm.utils.context import get_context
            modules = [m for m in llm.model_runner.model.modules() if isinstance(m, LinearBase)]
            expected = 4 * llm.model_runner.config.hf_config.num_hidden_layers
            if len(modules) != expected:
                raise ValueError("Qwen linear layout changed: expected QKV/O/gate-up/down per layer")
            llm._h2_linear_restore, timings["prefill_linear_policy"] = install_prefill_linear_policy(
                modules, get_context, config["linear_policy"])
        return llm, timings
    except BaseException:
        close_scheduling_engine(llm)
        raise


def close_scheduling_engine(llm):
    if hasattr(llm, "_h2_linear_restore"):
        llm._h2_linear_restore()
        del llm._h2_linear_restore
    close_corrected_engine(llm)


def summarize_traces(traces):
    requests = [r for trace in traces for r in trace["requests"]]
    intervals = [gap for r in requests for gap in r["inter_token_ms"]]
    metrics = {}
    for name in ("ttft", "tpot", "e2e", "admission_delay", "scheduler_wait"):
        for label, q in (("p50", .5), ("p95", .95), ("p99", .99)):
            metrics[f"{name}_{label}_ms"] = quantile([r[f"{name}_ms"] for r in requests], q)
    for label, q in (("p50", .5), ("p95", .95), ("p99", .99)):
        metrics[f"itl_{label}_ms"] = quantile(intervals, q)
    elapsed = sum(t["wall_seconds"] for t in traces)
    decode = [s for t in traces for s in t["steps"] if not s["is_prefill"]]
    metrics.update(requests_completed=len(requests), output_tokens=sum(r["output_tokens"] for r in requests),
        prompt_tokens=sum(r["prompt_tokens"] for r in requests),
        output_tokens_per_second=sum(r["output_tokens"] for r in requests) / elapsed,
        requests_per_second=len(requests) / elapsed, total_wall_seconds=elapsed,
        itl_max_ms=max(intervals), peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in traces),
        decode_steps=len(decode), prefill_steps=sum(s["is_prefill"] for t in traces for s in t["steps"]),
        mean_decode_batch_size=sum(s["batch_size"] for s in decode) / len(decode))
    return metrics


def execute_trace(llm, case, config, base, policy, *, probe=None):
    """Probe callbacks use a logical clock; timed runs use absolute wall arrivals.

    probe(request, token_index, cpu_bf16_logits) may record or compare logits.
    It is never called by performance runs. Tokens are teacher-forced only when
    probe is supplied. RNG-dependent sampled outputs need not match across policies.
    """
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context

    reset_cache(llm)
    pending = list(case["requests"])
    active, finished, steps = {}, {}, []
    logical_ms = 0
    if probe is not None:
        llm.model_runner.kv_cache.fill_(float("nan"))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    while pending or active:
        elapsed_ms = logical_ms if probe is not None else (time.perf_counter() - started) * 1000
        count = admission_count(policy, len(active), config["max_active_requests"], pending, elapsed_ms)
        for _ in range(count):
            request = pending.pop(0)
            params = SamplingParams(**base["sampling"], max_tokens=request["output_tokens"])
            llm.add_request(request["prompt_token_ids"], params)
            seq = llm.scheduler.waiting[-1]
            active[seq.seq_id] = dict(request=request, seq=seq,
                admitted_ms=(time.perf_counter() - started) * 1000, scheduled_ms=None, tokens_ms=[])
        if len(active) > config["max_active_requests"]:
            raise AssertionError("Upstream per-step limit must not masquerade as a resident request cap")
        if not active:
            wait_ms = pending[0]["arrival_ms"] - elapsed_ms
            if probe is not None:
                logical_ms += max(0, wait_ms)
            elif wait_ms > 0:
                time.sleep(wait_ms / 1000)
            continue
        active_before = len(active)
        seqs, is_prefill = llm.scheduler.schedule()
        scheduled_ms = (time.perf_counter() - started) * 1000
        ids = [active[s.seq_id]["request"]["request_id"] for s in seqs]
        for seq in seqs:
            state = active[seq.seq_id]
            if state["scheduled_ms"] is None:
                state["scheduled_ms"] = scheduled_ms
            if is_prefill and (seq.num_cached_tokens or seq.num_scheduled_tokens != seq.num_prompt_tokens):
                raise AssertionError("No-prefix, no-chunked-prefill scheduling control changed")
        if probe is None:
            token_ids = llm.model_runner.call("run", seqs, is_prefill)
        else:
            runner = llm.model_runner
            tokens, positions = runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
            try:
                with torch.inference_mode():
                    logits = runner.run_model(tokens, positions, is_prefill).detach().cpu().contiguous()
                if logits.dtype != torch.bfloat16 or not torch.isfinite(logits).all():
                    raise AssertionError("Expected finite BF16 logits, including poisoned-cache checks")
                for seq, row in zip(seqs, logits):
                    probe(active[seq.seq_id]["request"], seq.num_completion_tokens, row)
            finally:
                reset_context()
            token_ids = [seq.prompt_token_ids[-1-seq.num_completion_tokens] for seq in seqs]
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        visible_ms = (time.perf_counter() - started) * 1000
        done = []
        for seq in seqs:
            state = active[seq.seq_id]
            state["tokens_ms"].append(visible_ms)
            if seq.is_finished:
                key = state["request"]["request_id"]
                finished[key] = active.pop(seq.seq_id)
                done.append(key)
        steps.append(dict(is_prefill=is_prefill, batch_size=len(seqs), active_before=active_before,
            request_ids=ids, completed_request_ids=done, admission_check_ms=elapsed_ms,
            model_start_ms=scheduled_ms, token_visible_ms=visible_ms))
        logical_ms += config["logical_validation_step_ms"]
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    assert_released(llm)
    rows = []
    for request in case["requests"]:
        state = finished[request["request_id"]]
        times = state["tokens_ms"]
        if len(times) != request["output_tokens"]:
            raise AssertionError("Request output budget changed")
        rows.append(dict(request_id=request["request_id"], prompt_tokens=len(request["prompt_token_ids"]),
            output_tokens=len(times), arrival_ms=request["arrival_ms"], admitted_ms=state["admitted_ms"],
            first_scheduled_ms=state["scheduled_ms"], token_visible_ms=times,
            admission_delay_ms=state["admitted_ms"] - request["arrival_ms"],
            scheduler_wait_ms=state["scheduled_ms"] - state["admitted_ms"],
            ttft_ms=times[0] - request["arrival_ms"], e2e_ms=times[-1] - request["arrival_ms"],
            tpot_ms=(times[-1] - times[0]) / (len(times) - 1),
            inter_token_ms=[b-a for a, b in zip(times, times[1:])], cached_prompt_tokens=0,
            output_token_ids_sha256=token_hash(state["seq"].completion_token_ids)))
    return dict(wall_seconds=wall, peak_allocated_bytes=peak, requests=rows, steps=steps,
        finite_logits_checked=probe is not None, released_state=dict(running=0, waiting=0, used_blocks=0, nonzero_refcounts=0))


def validate_trace(trace, case, config, policy, *, timing=True):
    requests = trace["requests"]
    if [r["request_id"] for r in requests] != [r["request_id"] for r in case["requests"]]:
        raise ValueError("Missing, duplicate or reordered request records")
    if (not math.isfinite(trace["wall_seconds"]) or trace["wall_seconds"] <= 0
            or trace["peak_allocated_bytes"] <= 0 or any(trace["released_state"].values())):
        raise ValueError("Invalid trace duration or leaked request/KV state")
    states, offset = {}, 0
    by_id = {r["request_id"]: r for r in requests}
    for step in trace["steps"]:
        ids, completed = step["request_ids"], step["completed_request_ids"]
        if (len(set(ids)) != len(ids) or step["batch_size"] != len(ids)
                or not set(completed) <= set(ids) or not 1 <= len(ids) <= config["max_active_requests"]
                or not math.isfinite(step["admission_check_ms"]) or not math.isfinite(step["model_start_ms"])
                or not step["model_start_ms"] <= step["token_visible_ms"]):
            raise ValueError("Invalid scheduled batch")
        due = admission_count(policy, len(states), config["max_active_requests"],
            case["requests"][offset:], step["admission_check_ms"])
        if due != (len(ids) if step["is_prefill"] else 0):
            raise ValueError("Admission policy skipped eligible requests or admitted too early")
        if step["is_prefill"]:
            if policy == "fixed_cohort" and states:
                raise ValueError("Fixed cohort was refilled before the whole cohort completed")
            if ids != [r["request_id"] for r in requests[offset:offset+len(ids)]] or set(ids) & set(states):
                raise ValueError("Admission order changed or a request was admitted twice")
            for key in ids:
                states[key] = 0
            offset += len(ids)
        elif set(ids) != set(states):
            raise ValueError("Decode omitted a resident request")
        if len(states) != step["active_before"] or len(states) > config["max_active_requests"]:
            raise ValueError("Active request capacity changed")
        for key in ids:
            if states[key] == 0 and by_id[key]["first_scheduled_ms"] != step["model_start_ms"]:
                raise ValueError("First-schedule timestamp changed")
            states[key] += 1
            if (key in completed) != (states[key] == by_id[key]["output_tokens"]):
                raise ValueError("Request was released at an incorrect output boundary")
            if abs(by_id[key]["token_visible_ms"][states[key]-1] - step["token_visible_ms"]) > 1e-9:
                raise ValueError("Step timestamps differ from per-request timestamps")
        for key in completed:
            del states[key]
    if states or offset != len(requests):
        raise ValueError("Trace did not complete all requests")
    for actual, expected in zip(requests, case["requests"]):
        times = actual["token_visible_ms"]
        if (actual["output_tokens"] != expected["output_tokens"] or len(times) != expected["output_tokens"]
                or actual["prompt_tokens"] != len(expected["prompt_token_ids"])
                or actual["arrival_ms"] != expected["arrival_ms"] or actual["cached_prompt_tokens"]
                or len(actual["output_token_ids_sha256"]) != 64):
            raise ValueError("Per-request budget, arrival, or prefix contract changed")
        if not timing:
            continue
        if (not expected["arrival_ms"] <= actual["admitted_ms"] <= actual["first_scheduled_ms"] <= times[0]
                or any(not math.isfinite(t) for t in times) or times != sorted(times)
                or times[-1] > trace["wall_seconds"] * 1000 + 1e-6):
            raise ValueError("Per-request timing contract changed")
        values = dict(admission_delay_ms=actual["admitted_ms"] - expected["arrival_ms"],
            scheduler_wait_ms=actual["first_scheduled_ms"] - actual["admitted_ms"],
            ttft_ms=times[0] - expected["arrival_ms"], e2e_ms=times[-1] - expected["arrival_ms"],
            tpot_ms=(times[-1]-times[0]) / (len(times)-1))
        if any(not math.isfinite(actual[k]) or abs(actual[k]-v) > 1e-9 for k, v in values.items()):
            raise ValueError("Latency formula changed")
        if actual["inter_token_ms"] != [b-a for a, b in zip(times, times[1:])]:
            raise ValueError("ITL samples differ from actual token intervals")

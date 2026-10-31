"""Real-model KV-funded cohorts, with independent trace/accounting validation."""

from __future__ import annotations

import copy
import json
import math
import time

from serving.backends.nanovllm import ROOT, digest
from serving.h2_kv_capacity import static_window_manager
from serving.h2_prefix_probe import prefix_disabled_manager
from serving.h2_runtime import assert_released
from serving.h2_scheduling import create_scheduling_engine, summarize_traces, validate_trace
from serving.h2_workloads import token_hash
from serving.verify_h2_kv_capacity import load_inputs as load_capacity_inputs

DEFAULT_CONFIG = ROOT / "config/serving/h2_kv_pressure.json"
POLICIES = ("static_window", "paged")
SOURCES = ("serving/h2_kv_pressure.py", "serving/benchmark_h2_kv_pressure.py",
    "scripts/run_h2_kv_pressure.py", "scripts/summarize_h2_kv_pressure.py")


def request_credit(request, policy, *, block_size=256, window=1024):
    if policy not in POLICIES:
        raise ValueError("Unknown KV reservation policy")
    prompt, output = len(request["prompt_token_ids"]), request["output_tokens"]
    if not prompt or output < 2 or prompt + output > window or window % block_size:
        raise ValueError("Invalid request output/window budget")
    return window // block_size if policy == "static_window" else math.ceil((prompt+output-1)/block_size)


def funded_cohort(pending, policy, blocks, resident_limit=8):
    result, committed = [], 0
    for request in pending[:resident_limit]:
        needed = request_credit(request, policy)
        if committed + needed > blocks:
            break  # Strict FCFS: no skipping a large head request.
        result.append(request)
        committed += needed
    if pending and not result:
        raise ValueError("The head request cannot be funded; never enter a decode deadlock")
    return result, committed


def build_cases(config, prompts):
    expected = [dict(case_id="tight_short", num_blocks=16, output_tokens=64),
        dict(case_id="tight_growth", num_blocks=16, output_tokens=132),
        dict(case_id="ample_short", num_blocks=32, output_tokens=64)]
    if (config["schema_version"] != 1 or config["policies"] != list(POLICIES)
            or config["max_resident_requests"] != 8 or config["requests_per_trace"] != 16
            or config["independent_runs"] != 3 or config["measured_traces"] != 3 or config["warmup_traces"] != 1
            or config["admission_policy"] != "fcfs_fixed_cohort_peak_kv_funded"
            or config["require_exact_logits"] is not True or config["cases"] != expected
            or len(prompts) != 16 or any(len(p) != 128 for p in prompts)):
        raise ValueError("Qualify a fresh scope for different KV-pressure cases or controls")
    cases = {}
    for dimensions in expected:
        case = {**dimensions, "requests": [dict(request_id=f"r{i:02}", prompt_token_ids=p,
            output_tokens=dimensions["output_tokens"], arrival_ms=0) for i, p in enumerate(prompts)]}
        case["workload_sha256"] = token_hash(case)
        cases[case["case_id"]] = case
    return cases


def load_inputs(path=DEFAULT_CONFIG):
    config = json.loads(path.read_text())
    capacity, inputs, previous, _ = load_capacity_inputs(ROOT / config["base_capacity_config"])
    audit_path = ROOT / capacity["results_dir"] / "audit.json"
    audit = json.loads(audit_path.read_text())
    if (digest(audit_path) != config["base_capacity_audit_sha256"] or audit["status"] != "pass"
            or audit["provenance"] != previous or audit["full_model_exact_vectors"] != 1056):
        raise ValueError("Qualified KV allocator/source/model/environment changed")
    prompts = [r["prompt_token_ids"] for r in inputs[5]["equal_burst"]["requests"]]
    cases = build_cases(config, prompts)
    provenance = {**previous, "pressure_config_sha256": digest(path),
        "pressure_source_sha256": {name: digest(ROOT / name) for name in SOURCES},
        "pressure_base_capacity_audit_sha256": digest(audit_path)}
    return config, inputs, cases, provenance


def effective_pilot(pilot, case):
    result = copy.deepcopy(pilot)
    result["engine"]["num_kvcache_blocks"] = case["num_blocks"]
    return result


def engine_contract(pilot, case):
    return {**effective_pilot(pilot, case)["engine"], "enforce_eager": False,
        "prefix_cache": False, "dtype": "bfloat16", "common_max_resident_requests": 8}


def create_engine(inputs, case):
    scheduling, validation, pilot, _, h1, _, _ = inputs
    return create_scheduling_engine(h1, effective_pilot(pilot, case), validation, scheduling)


def reset_manager(llm, policy):
    from nanovllm.engine.block_manager import BlockManager
    assert_released(llm)
    cls = static_window_manager(BlockManager, 1024) if policy == "static_window" else prefix_disabled_manager(BlockManager)
    llm.scheduler.block_manager = cls(llm._h2_num_blocks, 256)


def summarize_pressure_traces(traces):
    result = summarize_traces(traces)
    result.update(max_resident_requests=max(s["batch_size"] for t in traces for s in t["steps"]),
        peak_used_kv_blocks=max(s["used_kv_blocks"] for t in traces for s in t["steps"]),
        peak_committed_kv_blocks=max(s["committed_kv_blocks"] for t in traces for s in t["steps"]))
    return result


def execute_trace(llm, case, config, base, policy, *, probe=None):
    """Same fixed-cohort scheduler on both sides; ONLY KV funding limits admission.

    Forwards use the real upstream scheduler/model/sampler. Probe CPU logit copies
    and deterministic teacher tokens are excluded from performance evidence.
    """
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context

    reset_manager(llm, policy)
    if probe is not None:
        llm.model_runner.kv_cache.fill_(float("nan"))
    pending, finished, steps = list(case["requests"]), {}, []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    while pending:
        batch, committed = funded_cohort(pending, policy, case["num_blocks"], config["max_resident_requests"])
        pending = pending[len(batch):]
        active, sequences = {}, []
        for request in batch:
            llm.add_request(request["prompt_token_ids"], SamplingParams(**base["sampling"], max_tokens=request["output_tokens"]))
            seq = llm.scheduler.waiting[-1]
            sequences.append(seq)
            active[seq.seq_id] = dict(request=request, seq=seq, admitted_ms=(time.perf_counter()-started)*1000,
                scheduled_ms=None, tokens_ms=[])
        step_index = 0
        while not llm.is_finished():
            check_ms = (time.perf_counter()-started)*1000
            seqs, prefill = llm.scheduler.schedule()
            scheduled_ms = (time.perf_counter()-started)*1000
            if seqs != sequences or prefill != (step_index == 0):
                raise AssertionError("Expected complete equal-length cohort, without chunking/preemption")
            manager = llm.scheduler.block_manager
            used = len(manager.used_block_ids)
            cached = [s.num_cached_tokens+s.num_scheduled_tokens for s in seqs]
            if used > committed or committed > case["num_blocks"]:
                raise AssertionError("Actual KV use exceeds the funded peak-growth budget")
            for seq in seqs:
                state = active[seq.seq_id]
                if state["scheduled_ms"] is None:
                    state["scheduled_ms"] = scheduled_ms
                if prefill and (seq.num_cached_tokens or seq.num_scheduled_tokens != 128):
                    raise AssertionError("No-prefix bulk Prefill changed")
            if probe is None:
                tokens = llm.model_runner.call("run", seqs, prefill)
            else:
                runner = llm.model_runner
                ids, positions = runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
                try:
                    with torch.inference_mode():
                        logits = runner.run_model(ids, positions, prefill).detach().cpu().contiguous()
                    if logits.dtype != torch.bfloat16 or not bool(torch.isfinite(logits).all()):
                        raise AssertionError("Nonfinite/non-BF16 poisoned-cache logits")
                    for seq, row in zip(seqs, logits):
                        probe(active[seq.seq_id]["request"], seq.num_completion_tokens, row)
                finally:
                    reset_context()
                tokens = [s.prompt_token_ids[-1-(s.num_completion_tokens % 128)] for s in seqs]
            llm.scheduler.postprocess(seqs, tokens, prefill)
            visible_ms = (time.perf_counter()-started)*1000
            keys, done = [], []
            for seq in seqs:
                state = active[seq.seq_id]
                key = state["request"]["request_id"]
                keys.append(key)
                state["tokens_ms"].append(visible_ms)
                if seq.is_finished:
                    finished[key] = state
                    done.append(key)
            steps.append(dict(is_prefill=prefill, request_ids=keys, batch_size=len(seqs), active_before=len(seqs),
                completed_request_ids=done, admission_check_ms=check_ms, model_start_ms=scheduled_ms,
                token_visible_ms=visible_ms, used_kv_blocks=used, free_kv_blocks=case["num_blocks"]-used,
                committed_kv_blocks=committed, cached_tokens_per_request=cached))
            step_index += 1
        assert_released(llm)
    torch.cuda.synchronize()
    wall, peak = time.perf_counter()-started, torch.cuda.max_memory_allocated()
    rows = []
    for request in case["requests"]:
        state, key = finished[request["request_id"]], request["request_id"]
        times = state["tokens_ms"]
        if len(times) != request["output_tokens"]:
            raise AssertionError("Output budget changed")
        rows.append(dict(request_id=key, prompt_tokens=128, output_tokens=len(times), arrival_ms=0,
            admitted_ms=state["admitted_ms"], first_scheduled_ms=state["scheduled_ms"], token_visible_ms=times,
            admission_delay_ms=state["admitted_ms"], scheduler_wait_ms=state["scheduled_ms"]-state["admitted_ms"],
            ttft_ms=times[0], e2e_ms=times[-1], tpot_ms=(times[-1]-times[0])/(len(times)-1),
            inter_token_ms=[b-a for a,b in zip(times,times[1:])], cached_prompt_tokens=0,
            output_token_ids_sha256=token_hash(state["seq"].completion_token_ids)))
    return dict(wall_seconds=wall, peak_allocated_bytes=peak, requests=rows, steps=steps,
        finite_logits_checked=probe is not None, released_state=dict(running=0, waiting=0, used_blocks=0, nonzero_refcounts=0))


def validate_pressure_trace(trace, case, config, policy, *, timing=True):
    batch, _ = funded_cohort(case["requests"], policy, case["num_blocks"], config["max_resident_requests"])
    # The historical validator independently replays FCFS, token budgets and
    # timing formulas. The tighter cap here is DERIVED from KV credits, not tuned.
    validate_trace(trace, case, dict(max_active_requests=len(batch)), "fixed_cohort", timing=timing)
    progress, offset, cohort = {}, 0, []
    for step in trace["steps"]:
        if step["is_prefill"]:
            if progress:
                raise ValueError("Refill before funded cohort completed")
            cohort, committed = funded_cohort(case["requests"][offset:], policy, case["num_blocks"], config["max_resident_requests"])
            if step["request_ids"] != [r["request_id"] for r in cohort]:
                raise ValueError("KV-funded FCFS admission changed")
            progress = {r["request_id"]: 0 for r in cohort}
            offset += len(cohort)
        cached = [128+progress[key] for key in step["request_ids"]]
        expected_used = (4*len(cohort) if policy == "static_window" else sum(math.ceil(n/256) for n in cached))
        if (step["cached_tokens_per_request"] != cached or step["committed_kv_blocks"] != committed
                or step["used_kv_blocks"] != expected_used or step["free_kv_blocks"] != case["num_blocks"]-expected_used
                or not 0 < expected_used <= committed <= case["num_blocks"]):
            raise ValueError("KV growth/accounting does not match actual token positions and funded peak")
        for key in step["request_ids"]:
            progress[key] += 1
        for key in step["completed_request_ids"]:
            del progress[key]
    if progress or offset != len(case["requests"]):
        raise ValueError("Incomplete funded trace")

"""H2-only adapters: fixed physical KV pool and an isolated prefix-admission switch.

The pinned upstream checkout is never edited. Imports of GPU packages are lazy so
the contract and statistics helpers can also be tested in the training environment.
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from serving.backends.nanovllm import ROOT, digest, load_config, model_manifest, validate_environment
from serving.h2_prefix_probe import prefix_disabled_manager
from serving.h2_workloads import token_hash

DEFAULT_PILOT = ROOT / "config/serving/h2_pilot.json"
SOURCE_FILES = (
    "serving/h2_runtime.py", "serving/benchmark_nanovllm.py",
    "serving/verify_h2_preflight.py", "serving/backends/nanovllm.py",
    "serving/h2_prefix_probe.py", "serving/h2_workloads.py",
    "scripts/run_h2_nanovllm.py", "scripts/summarize_h2_nanovllm.py",
)


def load_inputs(path=DEFAULT_PILOT):
    pilot = json.loads(path.read_text())
    config = json.loads((ROOT / pilot["h2_config"]).read_text())
    h1 = load_config(ROOT / config["h1_config"])
    validate_environment(h1)
    for name in config["required_h1_results"]:
        if json.loads((ROOT / "results/h_nanovllm" / name).read_text())["status"] != "pass":
            raise ValueError(f"H1 prerequisite failed: {name}")
    manifest = json.loads((ROOT / "results/h_nanovllm/h2_workload_manifest.json").read_text())
    if digest(ROOT / pilot["h2_config"]) != manifest["config_sha256"]:
        raise ValueError("H2 configuration differs from the fixed workload")
    if digest(ROOT / "serving/h2_workloads.py") != manifest["builder_sha256"]:
        raise ValueError("Workload builder differs from the recorded source")
    if digest(ROOT / config["workload_path"]) != manifest["workload_file_sha256"]:
        raise ValueError("Fixed workload file changed")
    workloads = json.loads((ROOT / config["workload_path"]).read_text())
    if workloads["model"] != h1["model"]:
        raise ValueError("Workload model pin differs from H1")
    for case in workloads["cases"]:
        if token_hash({k: v for k, v in case.items() if k != "workload_sha256"}) != case["workload_sha256"]:
            raise ValueError("Case token arrays do not match their hash")
    return pilot, config, h1, {c["case_id"]: c for c in workloads["cases"]}


def provenance(pilot_path, pilot, config, h1):
    import torch
    model = model_manifest(h1)
    repo = ROOT / h1["upstream"]["path"]
    return {
        "upstream_commit": h1["upstream"]["commit"],
        "upstream_source_sha256": {str(p.relative_to(repo)): digest(p) for p in sorted(repo.rglob("*.py"))},
        "model_revision": h1["model"]["revision"],
        "model_sha256": model["files"]["model.safetensors"]["sha256"],
        "tokenizer_sha256": {name: model["files"][name]["sha256"] for name in
            ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")},
        "model_config_sha256": model["files"]["config.json"]["sha256"],
        "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
        "pilot_config_sha256": digest(pilot_path),
        "h2_config_sha256": digest(ROOT / pilot["h2_config"]),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], text=True)),
            "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_cuda": torch.version.cuda,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        },
    }


def check_capacity(engine, case):
    batch, prompt, output = [case[k] for k in ("batch_size", "prompt_length", "output_length")]
    block = engine["kvcache_block_size"]
    needed = batch * math.ceil((prompt + output - 1) / block)
    if engine["num_kvcache_blocks"] < needed:
        raise ValueError(f"Cold decode needs {needed} blocks, not {engine['num_kvcache_blocks']}")
    if engine["max_num_seqs"] < batch or engine["max_num_batched_tokens"] < batch * prompt:
        raise ValueError("Engine cannot admit the entire cold wave before publishing hashes")
    if engine["max_model_len"] < prompt + output:
        raise ValueError("Model window is too short")
    return needed


def create_engine(h1, pilot, switches):
    import torch
    from nanovllm import LLM
    import nanovllm.engine.llm_engine as engine_module
    from nanovllm.engine.model_runner import ModelRunner

    if pilot["engine"]["tensor_parallel_size"] != 1:
        raise ValueError("This H2 adapter is tested only for TP=1")
    timings = {}

    class FixedPoolRunner(ModelRunner):
        def warmup_model(self):
            started = time.perf_counter()
            super().warmup_model()
            torch.cuda.synchronize()
            timings["upstream_model_warmup_seconds"] = time.perf_counter() - started

        def allocate_kv_cache(self):
            # Upstream overwrites config.num_kvcache_blocks with a memory heuristic.
            # Bind exactly the configured tensor instead, keeping every attention
            # module, physical layout and subsequent kernel path unchanged.
            cfg, hf = self.config, self.config.hf_config
            heads = hf.num_key_value_heads // self.world_size
            dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
            shape = (2, hf.num_hidden_layers, cfg.num_kvcache_blocks, self.block_size, heads, dim)
            if hf.dtype != torch.bfloat16:
                raise ValueError("H2 requires BF16")
            required = math.prod(shape) * hf.dtype.itemsize
            free, total = torch.cuda.mem_get_info()
            if required + pilot["minimum_free_after_kv_bytes"] > free:
                raise RuntimeError("Insufficient GPU headroom for the fixed KV pool")
            if torch.cuda.memory_allocated() + required > total * cfg.gpu_memory_utilization:
                raise RuntimeError("Fixed KV pool exceeds the common GPU budget")
            self.kv_cache = torch.empty(shape, dtype=hf.dtype, device="cuda")
            layers = [m for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
            if len(layers) != hf.num_hidden_layers:
                raise AssertionError("Attention layer count changed")
            for index, module in enumerate(layers):
                module.k_cache, module.v_cache = self.kv_cache[0, index], self.kv_cache[1, index]
            timings["kv_tensor_bytes"] = required
            timings["free_bytes_before_kv"] = free

        def capture_cudagraph(self):
            started = time.perf_counter()
            super().capture_cudagraph()
            torch.cuda.synchronize()
            timings["graph_capture_seconds"] = time.perf_counter() - started

    original = engine_module.ModelRunner
    engine_module.ModelRunner = FixedPoolRunner
    started = time.perf_counter()
    try:
        llm = LLM(str(ROOT / h1["model"]["path"]), **pilot["engine"],
                  enforce_eager=switches["enforce_eager"])
    finally:
        engine_module.ModelRunner = original
    torch.cuda.synchronize()
    timings["initialization_seconds"] = time.perf_counter() - started
    timings.setdefault("graph_capture_seconds", 0.0)
    llm._h2_prefix_cache = switches["prefix_cache"]
    llm._h2_num_blocks = pilot["engine"]["num_kvcache_blocks"]

    def reject_preemption(seq):
        raise RuntimeError("Preemption invalidates the no-preemption H2 control")
    llm.scheduler.preempt = reject_preemption
    reset_cache(llm)
    return llm, timings


def close_engine(llm):
    atexit.unregister(llm.exit)
    llm.exit()


def assert_released(llm):
    manager = llm.scheduler.block_manager
    if (not llm.is_finished() or manager.used_block_ids
            or len(manager.free_block_ids) != llm._h2_num_blocks
            or any(block.ref_count for block in manager.blocks)):
        raise AssertionError("Request queues or KV references leaked")


def reset_cache(llm):
    from nanovllm.engine.block_manager import BlockManager
    assert_released(llm)
    cls = BlockManager if llm._h2_prefix_cache else prefix_disabled_manager(BlockManager)
    llm.scheduler.block_manager = cls(llm._h2_num_blocks, llm.model_runner.block_size)


def quantile(values, q):
    if not values:
        raise ValueError("Empty latency sample")
    items = sorted(values)
    position = (len(items) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return items[lo] + (items[hi] - items[lo]) * (position - lo)


def aggregate_waves(waves):
    requests = [request for wave in waves for request in wave["requests"]]
    result = {}
    for name in ("ttft", "tpot", "e2e"):
        for label, q in (("p50", .5), ("p95", .95), ("p99", .99)):
            result[f"{name}_{label}_ms"] = quantile([r[f"{name}_ms"] for r in requests], q)
    result.update(
        output_tokens_per_second=sum(r["output_tokens"] for r in requests) / sum(w["wall_seconds"] for w in waves),
        peak_allocated_bytes=max(w["peak_allocated_bytes"] for w in waves),
        requests_completed=len(requests),
        prompt_tokens=sum(r["prompt_tokens"] for r in requests),
        output_tokens=sum(r["output_tokens"] for r in requests),
        cached_prompt_tokens=sum(w["cached_prompt_tokens"] for w in waves),
    )
    return result


def compile_counts():
    from torch._dynamo.utils import counters
    return {"unique_graphs": counters["stats"]["unique_graphs"], "frames_total": counters["frames"]["total"]}


def execute_wave(llm, case, phase, config, *, forced_tokens=None):
    """Time host-visible tokens; alternatively capture logits with forced inputs.

    ModelRunner.run's sampler .tolist() synchronizes the GPU before it returns.
    Timing the completed postprocess therefore includes Python scheduling and the
    sampled-token transfer, without adding a CUDA event/synchronize to every step.
    Forced-token mode is correctness evidence only and never contributes timings.
    """
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context

    assert_released(llm)
    requests = case["waves"][phase]
    output_length = case["output_length"] if forced_tokens is None else forced_tokens
    params = SamplingParams(**config["sampling"], max_tokens=output_length)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for request in requests:
        llm.add_request(request["prompt_token_ids"], params)
    sequences = list(llm.scheduler.waiting)
    first, last, hits, logits_by_step = {}, {}, {}, []
    steps = 0
    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        if len(seqs) != len(requests) or is_prefill != (steps == 0):
            raise AssertionError("Expected one bulk prefill followed by unpreempted bulk decode")
        if is_prefill:
            hits = {s.seq_id: s.num_cached_tokens for s in seqs}
            if any(s.num_cached_tokens + s.num_scheduled_tokens != s.num_prompt_tokens for s in seqs):
                raise AssertionError("Unexpected chunked prefill")
        if forced_tokens is None:
            token_ids = llm.model_runner.call("run", seqs, is_prefill)
        else:
            runner = llm.model_runner
            inputs, positions = runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
            try:
                with torch.inference_mode():
                    logits = runner.run_model(inputs, positions, is_prefill)
                logits_by_step.append(logits.detach().float().cpu())
            finally:
                reset_context()
            token_ids = [seq.prompt_token_ids[-1 - steps] for seq in seqs]
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        timestamp = time.perf_counter() - started
        for seq in seqs:
            first.setdefault(seq.seq_id, timestamp)
            if seq.is_finished:
                last[seq.seq_id] = timestamp
        steps += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    assert_released(llm)
    expected_hit = case["expected_warm_cached_tokens_per_request"] if llm._h2_prefix_cache and phase == "warm" else 0
    if list(hits.values()) != [expected_hit] * len(requests):
        raise AssertionError(f"Unexpected {phase} prefix hits: {list(hits.values())}")
    rows = []
    for request, seq in zip(requests, sequences):
        n = seq.num_completion_tokens
        if n != output_length:
            raise AssertionError("Output token budget changed")
        ttft, e2e = first[seq.seq_id], last[seq.seq_id]
        rows.append({"request_id": request["request_id"], "prompt_tokens": seq.num_prompt_tokens,
            "output_tokens": n, "cached_prompt_tokens": hits[seq.seq_id],
            "ttft_ms": ttft * 1000, "e2e_ms": e2e * 1000,
            "tpot_ms": (e2e - ttft) * 1000 / (n - 1),
            "output_token_ids_sha256": token_hash(seq.completion_token_ids)})
    return {"wall_seconds": elapsed, "peak_allocated_bytes": peak,
            "cached_prompt_tokens": sum(hits.values()), "steps": steps,
            "requests": rows}, logits_by_step


def now():
    return datetime.now(timezone.utc).isoformat()

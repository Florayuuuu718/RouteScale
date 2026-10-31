"""Independent GPU allocation/address + boundary-logit probes, never timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_kv_capacity import PageAllocator, allocator_report, assert_conservation
from serving.h2_runtime import assert_released, now
from serving.h2_scheduling import (
    close_scheduling_engine, create_scheduling_engine, load_scheduling_inputs, scheduling_provenance,
)
from serving.h2_workloads import token_hash

DEFAULT_CONFIG = ROOT / "config/serving/h2_kv_capacity.json"
SOURCES = ("serving/h2_kv_capacity.py", "serving/verify_h2_kv_capacity.py", "scripts/run_h2_kv_capacity.py")


def load_inputs(path):
    cfg = json.loads(path.read_text())
    if (cfg["schema_version"] != 1 or [cfg[k] for k in ("num_blocks", "block_size", "max_model_len")] != [32, 256, 1024]
            or cfg["capacity_lengths"] != [128, 256, 257, 512, 768, 1024]
            or cfg["fragmentation"] != dict(initial_requests=16, initial_tokens=512, release_stride=2, new_tokens=768)
            or cfg["gpu_validation"] != dict(prompt_case_id="b8_p128_o64_none", output_tokens=132,
                policies=["static_window", "paged"], require_exact_logits=True)):
        raise ValueError("Qualify a separate scope for a different capacity experiment")
    base_path = ROOT / cfg["base_scheduling_config"]
    inputs = load_scheduling_inputs(base_path)
    scheduling, _, pilot, _, _, cases, _ = inputs
    gate_path = ROOT / scheduling["results_dir"] / "preflight.json"
    if digest(gate_path) != cfg["base_gate_sha256"]:
        raise ValueError("Scheduling qualification gate changed")
    gate = json.loads(gate_path.read_text())
    current = scheduling_provenance(base_path, inputs)
    if gate["status"] != "pass" or not gate["gpu_execution_allowed"] or current != gate["provenance"]:
        raise ValueError("Qualified scheduling source/model/environment changed")
    for key, expected in (("num_kvcache_blocks", 32), ("kvcache_block_size", 256), ("max_model_len", 1024), ("max_num_seqs", 8)):
        if pilot["engine"][key] != expected:
            raise ValueError("Engine differs from allocation budget")
    if scheduling["prompt_case_id"] != cfg["gpu_validation"]["prompt_case_id"]:
        raise ValueError("Validation prompt fixture changed")
    provenance = {**current, "capacity_config_sha256": digest(path),
        "capacity_source_sha256": {name: digest(ROOT / name) for name in SOURCES},
        "capacity_base_gate_sha256": digest(gate_path)}
    prompts = [r["prompt_token_ids"] for r in cases["equal_burst"]["requests"][:8]]
    workload = dict(prompts=prompts, output_tokens=132, teacher_forcing="cyclic reversed prompt per request")
    return cfg, inputs, provenance, workload


def check_gpu_pages(pool, allocator):
    """Write/read ALL elements of owned pages; free pages must remain NaN."""
    import torch
    state = assert_conservation(allocator)
    pool.fill_(float("nan"))
    manager, ownership = allocator.manager, {}
    for key, seq in allocator.live.items():
        pages = manager.reservations[seq.seq_id] if hasattr(manager, "reservations") else seq.block_table
        for page in pages:
            if page in ownership:
                raise AssertionError("Aliased physical page")
            ownership[page] = key + 1
            pool[:, :, page].fill_(key + 1)
    for page in range(pool.shape[2]):
        view = pool[:, :, page]
        okay = (view == ownership[page]).all() if page in ownership else torch.isnan(view).all()
        if not bool(okay):
            raise AssertionError("GPU page addressing or sentinel isolation failed")
    torch.cuda.synchronize()
    return dict(state=state, owned_pages={str(k): v for k, v in ownership.items()},
        all_owned_elements_verified=True, all_free_elements_nan=True)


def address_probe(llm, cfg, base, sequence, policy):
    import torch
    pool = llm.model_runner.kv_cache
    pointer, before = pool.data_ptr(), torch.cuda.memory_allocated()
    results = []
    for tokens in cfg["capacity_lengths"]:
        allocator = PageAllocator(base, sequence, 32, 256, 1024, static=policy == "static_window")
        key = 0
        while allocator.allocate(key, tokens):
            key += 1
        results.append(dict(case=f"homogeneous_{tokens}", **check_gpu_pages(pool, allocator)))
        for key in list(allocator.live):
            allocator.release(key)
        assert_conservation(allocator)
    if policy == "paged":
        allocator = PageAllocator(base, sequence, 32, 256, 1024)
        for key in range(16):
            if not allocator.allocate(key, 512):
                raise AssertionError("GPU hole-reuse population cannot fit")
        for key in range(0, 16, 2):
            allocator.release(key)
        if not allocator.allocate(100, 768):
            raise AssertionError("Scattered free pages could not be reused")
        pages = allocator.live[100].block_table
        if max(pages)-min(pages)+1 == len(pages):
            raise AssertionError("Hole reuse did not exercise noncontiguous pages")
        results.append(dict(case="hole_reuse", new_request_page_table=list(pages), **check_gpu_pages(pool, allocator)))
        for key in list(allocator.live):
            allocator.release(key)
        assert_conservation(allocator)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    if pool.data_ptr() != pointer or before != after:
        raise AssertionError("Logical allocation unexpectedly changed physical GPU allocation")
    return dict(evidence_class="real_gpu_tensor_address_probe", cases=results,
        kv_tensor_shape=list(pool.shape), kv_tensor_bytes=pool.numel()*pool.element_size(),
        same_tensor_pointer=True, allocated_bytes_before=before, allocated_bytes_after=after,
        limitations="Model loaded, but no model calls at the reported allocation-capacity boundary. Not validated maximum serving concurrency or throughput.")


def boundary_probe(llm, cfg, workload, allocator, reference):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context
    assert_released(llm)
    llm.scheduler.block_manager = allocator.manager
    llm.model_runner.kv_cache.fill_(float("nan"))
    count = workload["output_tokens"]
    for prompt in workload["prompts"]:
        llm.add_request(prompt, SamplingParams(temperature=.6, ignore_eos=True, max_tokens=count))
    sequences = list(llm.scheduler.waiting)
    rows, checkpoints, compared, exact, max_error = [], [], 0, 0, 0.0
    for step in range(count):
        seqs, prefill = llm.scheduler.schedule()
        if seqs != sequences or prefill != (step == 0):
            raise AssertionError("Expected eight requests and no chunking/preemption")
        tokens, positions = (llm.model_runner.prepare_prefill(seqs) if prefill
            else llm.model_runner.prepare_decode(seqs))
        try:
            with torch.inference_mode():
                logits = llm.model_runner.run_model(tokens, positions, prefill).detach().cpu().contiguous()
            if logits.dtype != torch.bfloat16 or not bool(torch.isfinite(logits).all()):
                raise AssertionError("Nonfinite/non-BF16 full-vocabulary logits from poisoned KV")
            rows.append(logits)
            if reference is not None:
                target = reference[step]
                if target.shape != logits.shape or target.dtype != logits.dtype:
                    raise ValueError("Reference logits shape/dtype changed")
                matches = (logits == target).all(dim=1)
                exact += int(matches.sum())
                compared += len(seqs)
                max_error = max(max_error, float((logits.float()-target.float()).abs().max()))
        finally:
            reset_context()
        cached = [s.num_cached_tokens+s.num_scheduled_tokens for s in seqs]
        if step in (0, 128, 129, 131):
            finite_cache = all(bool(torch.isfinite(llm.model_runner.kv_cache[:, :, page, :min(256, n-i*256)]).all())
                for seq, n in zip(seqs, cached) for i, page in enumerate(seq.block_table))
            if not finite_cache:
                raise AssertionError("Live KV positions were not fully written")
            checkpoints.append(dict(output_step=step+1, cached_tokens_per_request=cached,
                page_tables=[list(s.block_table) for s in seqs],
                used_blocks=len(llm.scheduler.block_manager.used_block_ids), valid_kv_finite=True))
        forced = [s.prompt_token_ids[-1-(step % len(s.prompt_token_ids))] for s in seqs]
        llm.scheduler.postprocess(seqs, forced, prefill)
    assert_released(llm)
    if reference is not None and (exact != count*8 or compared != count*8):
        raise AssertionError(f"Strict capacity layout gate failed: {exact}/{compared} exact, max_abs={max_error}")
    return dict(evidence_class="full_model_gpu_boundary_probe", requests=8, output_tokens_per_request=count,
        total_output_tokens=count*8, cached_tokens_at_last_forward=128+count-1,
        full_vocabulary_vectors_checked=count*8, comparisons=compared, exact_vectors=exact,
        max_absolute_error=max_error if reference is not None else None,
        finite_bf16_logits_all_steps=True, kv_poisoned_before_run=True, checkpoints=checkpoints,
        used_blocks_after_release=0, nonzero_refcounts_after_release=0,
        limitations="Teacher-forced functional check, not timing or generation-quality evidence."), rows


def run_probe(path, policy):
    import torch
    cfg, inputs, provenance, workload = load_inputs(path)
    scheduling, validation, pilot, _, h1, _, _ = inputs
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.sequence import Sequence
    directory = ROOT / cfg["results_dir"]
    output = directory / f"{policy}.json"
    references = ROOT / "runs/validation/h2_kv_capacity" / directory.name / "static_logits.pt"
    tensor_path = references if policy == "static_window" else references.with_name("paged_logits.pt")
    if output.exists() or tensor_path.exists():
        raise ValueError("Preserve existing evidence; choose a new result/reference directory")
    reference = None
    if policy == "paged":
        baseline = json.loads((directory / "static_window.json").read_text())
        if baseline["status"] != "pass" or baseline["provenance"] != provenance or baseline["workload_sha256"] != token_hash(workload):
            raise ValueError("Baseline differs from this run")
        if digest(references) != baseline["reference_sha256"]:
            raise ValueError("Baseline tensor checksum changed")
        reference = torch.load(references, weights_only=True, mmap=True, map_location="cpu")
        if reference.shape != (132, 8, 151936):
            raise ValueError("Unexpected full-vocabulary reference shape")
    torch.manual_seed(20261023)
    llm, initialization = create_scheduling_engine(h1, pilot, validation, scheduling)
    try:
        addresses = address_probe(llm, cfg, BlockManager, Sequence, policy)
        allocator = PageAllocator(BlockManager, Sequence, 32, 256, 1024, static=policy == "static_window")
        boundary, rows = boundary_probe(llm, cfg, workload, allocator, reference)
        references.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.stack(rows), tensor_path)
        free = torch.cuda.mem_get_info()[0]
        if free < pilot["minimum_free_after_kv_bytes"]:
            raise AssertionError("GPU headroom below the common safety budget")
        report = dict(schema_version=1, stage="H2", status="pass", policy=policy,
            evidence_class="gpu_kv_capacity_functional_probe", created_at_utc=now(), provenance=provenance,
            workload_sha256=token_hash(workload), initialization=initialization,
            engine=pilot["engine"], kv_tensor_bytes=addresses["kv_tensor_bytes"],
            address_probe=addresses, boundary_probe=boundary, free_bytes_after_run=free,
            reference_path=str(references.relative_to(ROOT)), reference_sha256=digest(references),
            logits_path=str(tensor_path.relative_to(ROOT)), logits_sha256=digest(tensor_path))
        if policy == "static_window":
            report["allocator_replay"] = allocator_report(cfg, BlockManager, Sequence)
        write_result(report, output)
        print(f"KV capacity {policy}: pass, {boundary['full_vocabulary_vectors_checked']} finite vectors", flush=True)
    finally:
        close_scheduling_engine(llm)


def audit(path):
    import torch
    cfg, _, provenance, workload = load_inputs(path)
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.sequence import Sequence
    directory = ROOT / cfg["results_dir"]
    reports = {p: json.loads((directory / f"{p}.json").read_text()) for p in ("static_window", "paged")}
    replay = allocator_report(cfg, BlockManager, Sequence)
    if reports["static_window"]["allocator_replay"] != replay:
        raise ValueError("CPU allocation report did not recompute exactly")
    for policy, report in reports.items():
        if (report["status"] != "pass" or report["policy"] != policy or report["provenance"] != provenance
                or report["workload_sha256"] != token_hash(workload) or report["kv_tensor_bytes"] != 939524096
                or digest(ROOT / report["reference_path"]) != report["reference_sha256"]
                or digest(ROOT / report["logits_path"]) != report["logits_sha256"]):
            raise ValueError("GPU report provenance or reference changed")
        address = report["address_probe"]
        if (address["allocated_bytes_before"] != address["allocated_bytes_after"] or not address["same_tensor_pointer"]
                or address["kv_tensor_shape"] != [2, 28, 32, 256, 8, 128]):
            raise ValueError("Physical GPU allocation changed across logical capacity probes")
        expected_cases = [f"homogeneous_{n}" for n in cfg["capacity_lengths"]] + (["hole_reuse"] if policy == "paged" else [])
        if [c["case"] for c in address["cases"]] != expected_cases:
            raise ValueError("Missing or duplicated GPU address case")
        for case in address["cases"]:
            if not case["all_owned_elements_verified"] or not case["all_free_elements_nan"]:
                raise ValueError("GPU page isolation check missing")
            if case["case"].startswith("homogeneous_"):
                tokens = int(case["case"].split("_")[-1])
                row = next(r for r in replay["capacity"] if r["policy"] == policy and r["live_kv_tokens_per_request"] == tokens)
                if case["state"] != row["at_capacity"]:
                    raise ValueError("GPU allocation capacity differs from upstream CPU replay")
        b = report["boundary_probe"]
        if (b["full_vocabulary_vectors_checked"] != 1056 or not b["finite_bf16_logits_all_steps"]
                or not b["kv_poisoned_before_run"] or b["used_blocks_after_release"] or b["nonzero_refcounts_after_release"]):
            raise ValueError("Full-model boundary/release checks missing")
        if [c["cached_tokens_per_request"] for c in b["checkpoints"]] != [[n]*8 for n in (128, 256, 257, 259)]:
            raise ValueError("Missing both sides of the 256-token KV boundary")
        if [c["used_blocks"] for c in b["checkpoints"]] != ([32]*4 if policy == "static_window" else [8, 8, 16, 16]):
            raise ValueError("Logical window reservation/growth does not match expectations")
    candidate = reports["paged"]["boundary_probe"]
    if candidate["comparisons"] != 1056 or candidate["exact_vectors"] != 1056 or candidate["max_absolute_error"] != 0:
        raise ValueError("All logits must be exact; no relaxed tolerance")
    tensors = [torch.load(ROOT / reports[p]["logits_path"], weights_only=True, mmap=True, map_location="cpu") for p in reports]
    if any(t.shape != (132, 8, 151936) or t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("Saved full-vocabulary tensor shape or dtype changed")
    recomputed_exact = 0
    for left, right in zip(*tensors):
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            raise ValueError("Nonfinite saved full-vocabulary logits")
        recomputed_exact += int((left == right).all(dim=1).sum())
    if recomputed_exact != 1056:
        raise ValueError("Independent saved-logit comparison failed")
    if replay["fragmentation"]["contiguous_exact"]["accepted"] or not replay["fragmentation"]["paged"]["accepted"]:
        raise ValueError("Hole-reuse negative/positive controls failed")
    report = dict(schema_version=1, stage="H2", status="pass", evidence_class="kv_capacity_evidence_audit",
        created_at_utc=now(), provenance=provenance,
        raw_sha256={f"{p}.json": digest(directory / f"{p}.json") for p in reports},
        cpu_replay_recomputed_exactly=True, gpu_address_capacity_matches_replay=True,
        full_model_exact_vectors=recomputed_exact, saved_full_vocabulary_logits_recompared=True,
        performance_execution_allowed=False, kv_tensor_bytes=939524096,
        capacity_table=replay["capacity"], fragmentation=replay["fragmentation"], growth=replay["growth"],
        limitations=cfg["limitations"])
    write_result(report, directory / "audit.json")
    print("KV capacity audit: pass (allocation/address/full-model checks; NO timing claim)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", choices=("static_window", "paged"))
    args = parser.parse_args()
    if args.policy:
        run_probe(args.config, args.policy)
    else:
        audit(args.config)


if __name__ == "__main__":
    main()

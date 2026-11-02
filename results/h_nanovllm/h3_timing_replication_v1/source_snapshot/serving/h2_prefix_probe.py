"""Probe actual upstream block metadata using fixed cold/warm request waves."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

from serving.backends.nanovllm import ROOT, digest, load_config, write_result


def prefix_disabled_manager(base):
    class NoPrefixBlockManager(base):
        def can_allocate(self, seq):
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
    return NoPrefixBlockManager


def probe_case(case, manager_class, sequence_class, *, enabled, block_size=256, num_blocks=128):
    sequence_class.block_size = block_size
    cls = manager_class if enabled else prefix_disabled_manager(manager_class)
    manager = cls(num_blocks, block_size)
    phases = {}
    for phase in ("cold", "warm"):
        sequences, cached = [], []
        peak_refs = 0
        for request in case["waves"][phase]:
            seq = sequence_class(request["prompt_token_ids"])
            hit = manager.can_allocate(seq)
            if hit < 0:
                raise ValueError("Metadata pool cannot admit the entire request wave")
            manager.allocate(seq, hit)
            sequences.append(seq)
            cached.append(seq.num_cached_tokens)
            peak_refs = max(peak_refs, max(b.ref_count for b in manager.blocks))
        # Bulk admission precedes publication, as with one batched Prefill call.
        allocated = len(manager.used_block_ids)
        for seq in sequences:
            seq.num_scheduled_tokens = len(seq) - seq.num_cached_tokens
            manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
        for seq in sequences:
            manager.deallocate(seq)
        if manager.used_block_ids or len(manager.free_block_ids) != num_blocks:
            raise AssertionError("Blocks leaked at the end of a wave")
        total = sum(len(s) for s in sequences)
        phases[phase] = {"request_count": len(sequences), "prompt_tokens": total,
            "cached_tokens_by_request": cached, "cached_tokens": sum(cached),
            "computed_prefill_tokens": total - sum(cached), "hit_rate": sum(cached) / total,
            "prefill_physical_blocks": allocated, "peak_refcount": peak_refs,
            "used_blocks_after_release": len(manager.used_block_ids)}
        expected = (case["expected_warm_cached_tokens_per_request"]
                    if enabled and phase == "warm" else 0)
        if cached != [expected] * len(sequences):
            raise AssertionError(f"Unexpected cache behavior for {case['case_id']}: {cached}")
    return {"case_id": case["case_id"], "workload_sha256": case["workload_sha256"],
            "prefix_cache": enabled, "phases": phases}


def main():
    config = json.loads((ROOT / "config/serving/h2_nanovllm.json").read_text())
    h1 = load_config(ROOT / config["h1_config"])
    package = ModuleType("nanovllm")
    package.__path__ = [str(ROOT / h1["upstream"]["path"] / "nanovllm")]
    sys.modules["nanovllm"] = package
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.sequence import Sequence
    workloads = json.loads((ROOT / config["workload_path"]).read_text())
    manifest = json.loads((ROOT / "results/h_nanovllm/h2_workload_manifest.json").read_text())
    if digest(ROOT / config["workload_path"]) != manifest["workload_file_sha256"]:
        raise ValueError("Fixed workload file has changed")
    rows = [probe_case(case, BlockManager, Sequence, enabled=enabled,
            block_size=config["block_size"], num_blocks=config["metadata_probe_blocks"])
        for case in workloads["cases"] for enabled in (False, True)]
    report = {"schema_version": 1, "stage": "H2", "status": "pass",
        "evidence_class": "cpu_metadata_probe", "upstream": h1["upstream"],
        "probe_sha256": digest(Path(__file__)),
        "block_manager_sha256": digest(ROOT / h1["upstream"]["path"] / "nanovllm/engine/block_manager.py"),
        "workload_file_sha256": manifest["workload_file_sha256"],
        "metadata_pool_blocks": config["metadata_probe_blocks"], "variants": rows,
        "limitations": "Prefill metadata only; no GPU KV data, Decode, timing or memory-performance claim."}
    output = ROOT / "results/h_nanovllm/h2_prefix_probe.json"
    write_result(report, output)
    print(f"H2 CPU prefix probe: {len(rows)} case/mode combinations passed -> {output}")


if __name__ == "__main__":
    main()

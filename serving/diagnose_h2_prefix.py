"""Locate the H2 BF16 prefix mismatch without changing its acceptance thresholds.

This is an instrumented numerical diagnostic, never a performance benchmark.
NaN-poisoned physical cache, per-layer exact prefix checks and an FP32 attention
intervention distinguish page/reuse errors from computation-shape sensitivity.
"""

from __future__ import annotations

import json
from pathlib import Path
import traceback
from types import MethodType

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_runtime import (
    DEFAULT_PILOT, close_engine, create_engine, execute_wave, load_inputs, now, provenance, reset_cache,
)


def main():
    pilot, config, h1, cases = load_inputs()
    case = cases[pilot["correctness_case_id"]]
    directory = ROOT / pilot["results_dir"]
    report = {"schema_version": 1, "stage": "H2", "status": "running",
        "evidence_class": "instrumented_numerical_diagnostic", "created_at_utc": now(),
        "diagnostic_sha256": digest(Path(__file__)), "does_not_release_formal_gate": True,
        "original_gate_sha256": digest(directory / "preflight.json"),
        "original_tolerances": {k: v for k, v in pilot.items() if k.startswith("correctness_")}}
    llm = None
    try:
        import torch
        from nanovllm.layers.attention import store_kvcache
        from nanovllm.utils.context import get_context
        from torch.nn.attention import sdpa_kernel, SDPBackend
        report["provenance"] = provenance(DEFAULT_PILOT, pilot, config, h1)
        llm, report["initialization"] = create_engine(h1, pilot, config["variants"]["eager_no_prefix"])
        modules = [layer.self_attn.attn for layer in llm.model_runner.model.model.layers]
        originals = [module.forward for module in modules]
        state = {"phase": None, "capture": False, "rows": [], "prefix": {}, "prefix_checks": []}

        def last_tokens(value, lengths):
            return torch.stack([value[index - 1] for index in lengths]).detach().float().cpu()

        def hook(index):
            def record(module, inputs, output):
                context = get_context()
                if not state["capture"] or not context.is_prefill:
                    return
                seqs = list(llm.scheduler.running)
                prefix = torch.stack((module.k_cache[seqs[-1].block_table[0], :256],
                                      module.v_cache[seqs[-1].block_table[0], :256]))
                if state["phase"] == "cold":
                    state["prefix"][index] = prefix.clone()
                else:
                    # For no-prefix this is the recomputed prefix; for prefix-on
                    # all requests share the physical resurrected cold block.
                    state["prefix_checks"].append(torch.equal(prefix, state["prefix"][index]))
                q, k, v = inputs
                lengths = context.cu_seqlens_q[1:].tolist()
                state["rows"].append({"layer": index, "phase": state["phase"],
                    "q": last_tokens(q, lengths), "k": last_tokens(k, lengths),
                    "v": last_tokens(v, lengths), "attention_out": last_tokens(output, lengths)})
            return record

        handles = [module.register_forward_hook(hook(index)) for index, module in enumerate(modules)]

        def reference_attention(module, q, k, v):
            context = get_context()
            store_kvcache(k, v, module.k_cache, module.v_cache, context.slot_mapping)
            seqs = list(llm.scheduler.running)
            # Read scheduled order directly: for decode it is all running seqs in
            # their original order; this probe rejects partial/mixed admission.
            queries = q.split([s.num_scheduled_tokens for s in seqs])
            outputs = []
            for seq, query in zip(seqs, queries):
                length = seq.num_cached_tokens + seq.num_scheduled_tokens
                key = module.k_cache[seq.block_table].flatten(0, 1)[:length]
                value = module.v_cache[seq.block_table].flatten(0, 1)[:length]
                ratio = q.size(1) // key.size(1)
                positions = torch.arange(query.size(0), device=q.device) + seq.num_cached_tokens
                mask = torch.arange(length, device=q.device)[None, :] <= positions[:, None]
                with sdpa_kernel(SDPBackend.MATH):
                    out = torch.nn.functional.scaled_dot_product_attention(
                        query.transpose(0, 1).unsqueeze(0).float(),
                        key.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0).float(),
                        value.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0).float(),
                        attn_mask=mask, scale=module.scale)
                outputs.append(out.squeeze(0).transpose(0, 1).to(q.dtype))
            return torch.cat(outputs)

        def run(enabled, *, capture=True):
            llm._h2_prefix_cache = enabled
            reset_cache(llm)
            llm.model_runner.kv_cache.fill_(float("nan"))
            state.update(capture=capture, rows=[], prefix={}, prefix_checks=[])
            logits = {}
            for phase in ("cold", "warm"):
                state["phase"] = phase
                _, logits[phase] = execute_wave(llm, case, phase, config,
                    forced_tokens=pilot["correctness_output_tokens"])
                if any(not torch.isfinite(x).all() for x in logits[phase]):
                    raise AssertionError("NaN-poisoned cache reached model logits")
            if capture and state["prefix_checks"] != [True] * len(modules):
                raise AssertionError("Recomputed/reused prefix differs from the saved cold prefix")
            return logits, state["rows"], len(state["prefix_checks"])

        def metrics(expected, actual):
            delta = actual - expected
            return {"max_abs_error": delta.abs().max().item(),
                "relative_rmse": (delta.square().mean() / expected.square().mean()).sqrt().item(),
                "top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).float().mean().item(),
                "original_elementwise_tolerance_pass": torch.allclose(actual, expected,
                    atol=pilot["correctness_atol"], rtol=pilot["correctness_rtol"])}

        report["backends"] = {}
        for backend in ("upstream_flash_attention", "fp32_sdpa_attention_only"):
            if backend.startswith("fp32"):
                for module in modules:
                    module.forward = MethodType(reference_attention, module)
            baseline, base_rows, base_checks = run(False)
            repeat, _, _ = run(False, capture=False)
            candidate, changed_rows, changed_checks = run(True)
            comparisons = [{"phase": phase, "step": step,
                **metrics(expected, actual)} for phase in ("cold", "warm")
                for step, (expected, actual) in enumerate(zip(baseline[phase], candidate[phase]))]
            aa_exact = all(torch.equal(a, b) for phase in baseline for a, b in zip(baseline[phase], repeat[phase]))
            if not aa_exact:
                raise AssertionError("Same-path A/A teacher-forced logits were nondeterministic")
            layers = []
            for before, after in zip(base_rows, changed_rows):
                layers.append({"layer": before["layer"], "phase": before["phase"],
                    **{f"{key}_max_abs_error": (after[key] - before[key]).abs().max().item()
                       for key in ("q", "k", "v", "attention_out")}})
            report["backends"][backend] = {"aa_exact": aa_exact,
                "cold_prefix_matches_warm_recomputed_layers": base_checks,
                "cold_prefix_matches_warm_reused_layers": changed_checks,
                "nan_poisoned_cache_finite_logits": True,
                "comparisons": comparisons, "layers": layers}
            print(backend, "warm max error:", max(r["max_abs_error"] for r in comparisons), flush=True)
        for module, original in zip(modules, originals):
            module.forward = original
        for handle in handles:
            handle.remove()
        report.update(status="pass",
            limitations="Diagnostic completion only: BF16 GEMM/norm remain unchanged even with FP32 attention. No threshold was relaxed; original formal gate remains failed.")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                close_engine(llm)
        finally:
            report["finished_at_utc"] = now()
            write_result(report, directory / "prefix_numerical_diagnostic.json")


if __name__ == "__main__":
    main()

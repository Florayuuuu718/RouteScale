"""Cold Prefill batch-shape controls, not a benchmark or a gate release."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_corrected_runtime import close_corrected_engine, create_corrected_engine
from serving.h2_runtime import assert_released, compile_counts, now, reset_cache
from serving.h2_scheduling import DEFAULT_SCHEDULING, SWITCHES, load_scheduling_inputs, scheduling_provenance


def difference(a, b):
    import torch
    a, b = a.contiguous(), b.contiguous()
    delta = a.float() - b.float()
    return dict(exact=torch.equal(a, b), max_absolute_error=delta.abs().max().item(),
        relative_rmse=(delta.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-12)).item())


def main():
    path = ROOT / "results/h_nanovllm/h2_scheduling/shape_diagnostic.json"
    if path.exists():
        raise SystemExit("Do not overwrite diagnostic evidence")
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = load_scheduling_inputs()
    config, validation, pilot, base, h1, cases, _ = inputs
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context
    original_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    report = dict(schema_version=1, stage="H2", status="running", evidence_class="scheduling_shape_diagnostic",
        created_at_utc=now(), source_sha256=digest(Path(__file__)), original_gate_sha256=digest(path.parent / "preflight.json"),
        provenance=scheduling_provenance(DEFAULT_SCHEDULING, inputs), controls={},
        does_not_release_formal_gate=True, original_bf16_reduced_precision_reduction=original_flag)
    llm, handles = None, []
    try:
        llm, report["initialization"] = create_corrected_engine(h1, pilot, SWITCHES, validation)
        layer = llm.model_runner.model.model.layers[0]
        snapshots = {}

        def capture(name):
            def hook(module, args, output):
                if name == "qkv":
                    snapshots["normalized_hidden"] = args[0].detach().cpu().clone()
                if name == "attention":
                    for key, tensor in zip(("q", "k", "v"), args):
                        snapshots[key] = tensor.detach().cpu().clone()
                snapshots[name] = output.detach().cpu().clone()
            return hook
        handles = [layer.self_attn.qkv_proj.register_forward_hook(capture("qkv")),
            layer.self_attn.attn.register_forward_hook(capture("attention")),
            layer.self_attn.o_proj.register_forward_hook(capture("o_proj"))]
        requests = cases["mixed_burst"]["requests"][8:]

        def prefill(selected):
            reset_cache(llm)
            llm.model_runner.kv_cache.fill_(float("nan"))
            snapshots.clear()
            for request in selected:
                llm.add_request(request["prompt_token_ids"], SamplingParams(**base["sampling"], max_tokens=1))
            seqs, is_prefill = llm.scheduler.schedule()
            if not is_prefill or len(seqs) != len(selected):
                raise ValueError("Cold control did not admit the entire requested batch")
            runner = llm.model_runner
            tokens, positions = runner.prepare_prefill(seqs)
            try:
                with torch.inference_mode():
                    logits = runner.run_model(tokens, positions, True).detach().cpu().clone()
                if logits.dtype != torch.bfloat16 or not torch.isfinite(logits).all():
                    raise ValueError("Cold BF16 control has nonfinite logits")
            finally:
                reset_context()
            llm.scheduler.postprocess(seqs, [s.prompt_token_ids[-1] for s in seqs], True)
            assert_released(llm)
            return logits, dict(snapshots)

        reference_metadata = json.loads((path.parent / "preflight/fixed_cohort.json").read_text())["cases"]["mixed_burst"]["reference_logits"]
        reference_path = ROOT / reference_metadata["path"]
        if digest(reference_path) != reference_metadata["sha256"]:
            raise ValueError("Saved reference changed")
        saved = torch.load(reference_path, weights_only=True, map_location="cpu")
        for label, flag in (("original", original_flag), ("no_reduced_precision_reduction", False)):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
            full, full_trace = prefill(requests)
            repeat, _ = prefill(requests)
            rows = {"batch8_AA": difference(full, repeat),
                "batch8_vs_saved": difference(full, torch.stack([saved[r["request_id"]][0] for r in requests])), "halves": []}
            for start in (0, 4):
                half, half_trace = prefill(requests[start:start+4])
                rows["halves"].append(dict(request_ids=[r["request_id"] for r in requests[start:start+4]],
                    logits=difference(half, full[start:start+4]),
                    first_layer={name: difference(half_trace[name], full_trace[name][start*128:(start+4)*128]) for name in half_trace}))
            report["controls"][label] = rows
            write_result(report, path)
        report.update(status="pass", compiler_settings=dict(recompile_limit=torch._dynamo.config.recompile_limit,
            fail_on_recompile_limit_hit=torch._dynamo.config.fail_on_recompile_limit_hit,
            suppress_errors=torch._dynamo.config.suppress_errors), compile_counters=compile_counts())
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_flag
        try:
            if llm is not None:
                close_corrected_engine(llm)
        finally:
            report["finished_at_utc"] = now()
            write_result(report, path)
    print(json.dumps(report["controls"], indent=2))


if __name__ == "__main__":
    main()

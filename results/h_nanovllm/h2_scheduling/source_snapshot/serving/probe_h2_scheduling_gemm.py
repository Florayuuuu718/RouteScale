"""Isolate BF16 linear row-shape rounding; tile controls are not a runtime fix."""

from __future__ import annotations

import fcntl
from pathlib import Path
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_corrected_runtime import close_corrected_engine, create_corrected_engine
from serving.h2_runtime import assert_released, now, reset_cache
from serving.h2_scheduling import DEFAULT_SCHEDULING, SWITCHES, load_scheduling_inputs, scheduling_provenance


def main():
    output = ROOT / "results/h_nanovllm/h2_scheduling/gemm_diagnostic.json"
    if output.exists():
        raise SystemExit("Do not overwrite diagnostics")
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = load_scheduling_inputs()
    config, validation, pilot, base, h1, cases, _ = inputs
    import torch
    import torch.nn.functional as F
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context
    report = dict(schema_version=1, stage="H2", status="running", evidence_class="isolated_linear_shape_diagnostic",
        created_at_utc=now(), source_sha256=digest(Path(__file__)),
        original_gate_sha256=digest(output.parent / "preflight.json"),
        provenance=scheduling_provenance(DEFAULT_SCHEDULING, inputs), does_not_release_formal_gate=True,
        limitations="One first-layer projection, not a full-model correction or performance measurement. The FP32 path is reference-only; tile512 is an isolated control, not installed in the runtime.")
    llm, handle, saved = None, None, {}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        llm, report["initialization"] = create_corrected_engine(h1, pilot, SWITCHES, validation)
        projection = llm.model_runner.model.model.layers[0].self_attn.o_proj

        def hook(module, args, value):
            saved["input"] = args[0].detach().clone()
            saved["output"] = value.detach().clone()
        handle = projection.register_forward_hook(hook)
        reset_cache(llm)
        llm.model_runner.kv_cache.fill_(float("nan"))
        for request in cases["mixed_burst"]["requests"][8:]:
            llm.add_request(request["prompt_token_ids"], SamplingParams(**base["sampling"], max_tokens=1))
        seqs, is_prefill = llm.scheduler.schedule()
        if not is_prefill or len(seqs) != 8:
            raise ValueError("Expected eight cold requests")
        runner = llm.model_runner
        tokens, positions = runner.prepare_prefill(seqs)
        try:
            with torch.inference_mode():
                logits = runner.run_model(tokens, positions, True)
            if not torch.isfinite(logits).all():
                raise ValueError("Poisoned-cache control has nonfinite logits")
        finally:
            reset_context()
        llm.scheduler.postprocess(seqs, [s.prompt_token_ids[-1] for s in seqs], True)
        assert_released(llm)
        handle.remove()
        handle = None

        def compare(a, b):
            a, b = a.detach().float().cpu(), b.detach().float().cpu()
            delta = a - b
            return dict(exact=torch.equal(a, b), unequal_elements=(a != b).sum().item(), elements=a.numel(),
                max_absolute_error=delta.abs().max().item(),
                relative_rmse=(delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)).item())

        with torch.inference_mode():
            x, weight = saved["input"], projection.weight
            full = F.linear(x, weight)
            halves = torch.cat([F.linear(part, weight) for part in x.split(512)], dim=0)
            # Both batch sizes execute the same non-padded 512-row tiles. This
            # tests the proposed shape rule only for this one captured linear.
            tiled_full = torch.cat([F.linear(x[i:i+512], weight) for i in range(0, len(x), 512)], dim=0)
            tiled_halves = torch.cat([torch.cat([F.linear(part[j:j+512], weight)
                for j in range(0, len(part), 512)]) for part in x.split(512)])
            torch.backends.cuda.matmul.allow_tf32 = False
            fp32 = F.linear(x.float(), weight.float())
            fp32_halves = torch.cat([F.linear(part.float(), weight.float()) for part in x.split(512)])
            report["comparisons"] = dict(
                native_projection_vs_replayed_full=compare(saved["output"], full),
                bf16_full1024_vs_two512=compare(full, halves),
                bf16_full1024_vs_rounded_fp32=compare(full, fp32.bfloat16()),
                bf16_two512_vs_rounded_fp32=compare(halves, fp32.bfloat16()),
                fp32_full_vs_two512=compare(fp32, fp32_halves),
                tile512_batch8_vs_batch4=compare(tiled_full, tiled_halves))
        report.update(status="pass", input_shape=list(x.shape), weight_shape=list(weight.shape),
            model_dtype=str(x.dtype), reference_tf32=False,
            compiler_settings=dict(recompile_limit=torch._dynamo.config.recompile_limit,
                fail_on_recompile_limit_hit=torch._dynamo.config.fail_on_recompile_limit_hit,
                suppress_errors=torch._dynamo.config.suppress_errors))
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        if handle is not None:
            handle.remove()
        if llm is not None:
            close_corrected_engine(llm)
        report["finished_at_utc"] = now()
        write_result(report, output)
    print(report["comparisons"], flush=True)


if __name__ == "__main__":
    main()

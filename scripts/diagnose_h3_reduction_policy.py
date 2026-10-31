"""Single-switch BF16 GEMM reduction probes; never permits performance runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_performance import load_inputs, load_native, create_path, run_wave, compare_tensors, tensor_metadata, now


def main():
    import torch
    import torch.nn.functional as F
    config_path = ROOT / "config/serving/h3_performance_v3.json"
    config, artifact, cases, provenance = load_inputs(config_path)
    output = ROOT / config["results_dir"] / "reduction_policy_diagnostic.json"
    if output.exists():
        raise ValueError("Preserve existing reduction diagnostic")
    original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    report = dict(schema_version=1, stage="H3", status="diagnostic", created_at_utc=now(),
        provenance=provenance, source_sha256=digest(Path(__file__)), original_flag=original, comparisons={})
    model = load_native(artifact)
    case = next(c for c in cases if c["case_id"] == "batch_short")
    old = json.loads((ROOT / "results/h_nanovllm/h3_performance_v1/reference_shape_diagnostic.json").read_text())
    step = old["failing_step"]
    inputs = []

    def capture(module, args):
        inputs.append(args[0].detach().clone())

    projection = model.transformer.h[0].mlp.c_proj
    hook = projection.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            context = torch.tensor([p + f[:step] for p, f in zip(case["prompt_ids"], case["continuation_ids"])], device="cuda")
            model(context)
    finally:
        hook.remove()
    same_input = inputs[0]
    report["projection_control"] = {}
    with torch.inference_mode():
        for allow in (True, False):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow
            batch = F.linear(same_input, projection.weight, projection.bias)
            slices = torch.cat([F.linear(row[None], projection.weight, projection.bias) for row in same_input])
            report["projection_control"][str(allow)] = dict(exact=torch.equal(batch, slices),
                max_absolute_error=float((batch.float() - slices.float()).abs().max()),
                differing_elements=int((batch != slices).sum()))
    del model, projection, same_input, inputs, batch, slices
    torch.cuda.empty_cache()
    reference = json.loads((ROOT / config["results_dir"] / "reference.json").read_text())
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for variant, case_id in (("native_kv", "page_growth"), ("paged_eager", "batch_short")):
            case = next(c for c in cases if c["case_id"] == case_id)
            path = create_path(artifact, case, config, variant)
            try:
                trace, actual, _ = run_wave(path, case, "model_execution", collect_logits=True, poison=True)
                metadata = tensor_metadata(actual, ROOT / config["validation_dir"] / f"reduction_off_{variant}_{case_id}.pt")
                expected = torch.load(ROOT / reference["cases"][case_id]["path"], mmap=True, weights_only=True, map_location="cpu")
                report["comparisons"][f"{variant}/{case_id}"] = dict(
                    **compare_tensors(actual, expected, config["tolerances"], fp32_reference=True),
                    tensor=metadata, trace=trace, bf16_weight_sha256=path.weight_sha256)
            finally:
                path.close()
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original
    report["limitations"] = "Two diagnostic cases, not a new gate or formal timing. Same BF16 weights/inputs/outputs; only GEMM intermediate reduction permission changes."
    write_result(report, output)
    print(json.dumps(dict(projection_control=report["projection_control"],
        comparisons={k: {a: b for a, b in v.items() if a not in ("tensor", "trace")} for k, v in report["comparisons"].items()}), indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Diagnose BF16 reference batch shape without changing an existing H3 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, load_config, validate_environment, write_result
from serving.h3_performance import compare_tensors, load_native, now, tensor_metadata


def diagnose(config_path):
    import torch
    config = json.loads(config_path.read_text())
    validate_environment(load_config(ROOT / config["h1_config"]))
    directory = ROOT / config["results_dir"]
    output = directory / "reference_shape_diagnostic.json"
    if output.exists():
        raise ValueError("Preserve existing diagnostic")
    preparation = json.loads((directory / "preparation.json").read_text())
    if preparation["workload_sha256"] != digest(ROOT / config["workload_path"]):
        raise ValueError("Frozen requests changed")
    gate = json.loads((directory / "gate.json").read_text())
    manifest = json.loads((ROOT / config["artifact_manifest"]).read_text())
    artifact = ROOT / manifest["artifact_path"]
    if digest(artifact) != config["artifact_sha256"]:
        raise ValueError("Artifact changed")
    case = next(c for c in json.loads((ROOT / config["workload_path"]).read_text())["cases"] if c["case_id"] == "batch_short")
    model = load_native(artifact)
    rows = []
    with torch.inference_mode():
        for step in range(case["output_length"]):
            rows.append(torch.cat([model(torch.tensor([prompt + forced[:step]], device="cuda"))[0][:, -1].cpu()
                for prompt, forced in zip(case["prompt_ids"], case["continuation_ids"])]))
    reference = torch.stack(rows)
    metadata = tensor_metadata(reference, ROOT / config["validation_dir"] / "diagnostic_single_request_reference.pt")
    reports = {"batched_reference": json.loads((directory / "reference.json").read_text())}
    reports.update({v: json.loads((directory / "qualification" / f"{v}.json").read_text()) for v in config["variants"]})
    comparisons, tensors = {}, {}
    for name, report in reports.items():
        entry = report["cases"][case["case_id"]]
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Saved tensor changed")
        actual = torch.load(ROOT / entry["path"], mmap=True, weights_only=True, map_location="cpu")
        comparisons[name] = compare_tensors(actual, reference, config["tolerances"])
        tensors[name] = actual
    a, b = tensors["paged_eager"].float(), tensors["batched_reference"].float()
    relative = (a - b).square().mean(-1).sqrt() / b.square().mean(-1).sqrt().clamp_min(1e-12)
    failing = torch.nonzero(relative > config["tolerances"]["max_relative_rmse"])
    step, request = failing[0].tolist()
    block = model.transformer.h[0]
    points = {"ln_1": block.ln_1, "qkv": block.attn.c_attn,
        "attention_projection": block.attn.c_proj, "ln_2": block.ln_2,
        "mlp_fc": block.mlp.c_fc, "mlp_projection": block.mlp.c_proj, "lm_head": model.lm_head}
    captures = {}
    hooks = []
    for name, module in points.items():
        def capture(module, args, out, name=name):
            captures.setdefault(name, []).append(out.detach().cpu().contiguous())
        hooks.append(module.register_forward_hook(capture))

    def projection_input(module, args):
        captures.setdefault("attention_input", []).append(args[0].detach().cpu().contiguous())

    hooks.append(block.attn.c_proj.register_forward_pre_hook(projection_input))
    contexts = [prompt + forced[:step] for prompt, forced in zip(case["prompt_ids"], case["continuation_ids"])]
    try:
        with torch.inference_mode():
            model(torch.tensor(contexts, device="cuda"))
            batched = captures
            captures = {}
            for context in contexts:
                model(torch.tensor([context], device="cuda"))
            single = captures
    finally:
        for hook in hooks:
            hook.remove()
    layer_checks = {}
    for name in ("ln_1", "qkv", "attention_input", "attention_projection", "ln_2", "mlp_fc", "mlp_projection", "lm_head"):
        left, right = batched[name][0], torch.cat(single[name], dim=0)
        layer_checks[name] = dict(exact=torch.equal(left, right),
            max_absolute_error=float((left.float() - right.float()).abs().max()),
            differing_elements=int((left != right).sum()), shape=list(left.shape))
    report = dict(schema_version=1, stage="H3", status="diagnostic", created_at_utc=now(),
        evidence_class="bf16_reference_batch_shape_diagnostic", source_sha256=digest(Path(__file__)),
        config_sha256=digest(config_path), failed_gate_sha256=digest(directory / "gate.json"),
        original_failure_unchanged=gate["status"] == "fail", reference=metadata, comparisons=comparisons,
        failing_step=step, failing_request=request, first_block_checks=layer_checks,
        first_differing_probe=next(name for name, value in layer_checks.items() if not value["exact"]),
        limitations="Reference-shape intervention only. Does not repair a runtime kernel, change thresholds "
        "or turn the failed batched-reference gate into pass. A canonical-reference scope requires a fresh gate.")
    write_result(report, output)
    print(json.dumps(dict(comparisons=comparisons, first_differing_probe=report["first_differing_probe"]), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/serving/h3_performance_v1.json")
    diagnose(parser.parse_args().config)

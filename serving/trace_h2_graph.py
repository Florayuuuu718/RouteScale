"""Instrument captured/eager Decode intermediates to locate batch-1 divergence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_numerical_policy import install_shape_policy
from serving.h2_runtime import DEFAULT_PILOT, close_engine, create_engine, execute_wave, load_inputs, now, reset_cache
from serving.probe_h2_numerics import compare_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "results/h_nanovllm/h2_numerics/graph_layer_trace.json")
    args = parser.parse_args()
    if args.result.exists():
        raise SystemExit("Select a new output to preserve diagnostic evidence")
    pilot, config, h1, cases = load_inputs()
    import torch
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.utils.context import get_context

    report = {"schema_version": 1, "stage": "H2", "status": "running", "created_at_utc": now(),
        "evidence_class": "instrumented_graph_numerical_diagnostic", "gpu_execution_allowed": False,
        "entrypoint_sha256": digest(Path(__file__)), "modes": {}}
    buffers, handles = {}, []

    def store(name, values):
        if get_context().is_prefill:
            return
        values = values if isinstance(values, tuple) else (values,)
        for index, tensor in enumerate(values):
            if tensor is None:
                continue
            key = f"{name}/{index}"
            value = tensor[0].flatten()
            if key not in buffers:
                buffers[key] = torch.empty(value.shape, device="cuda", dtype=torch.float32)
            buffers[key].copy_(value)

    original_capture = ModelRunner.capture_cudagraph
    def traced_capture(self):
        for index, layer in enumerate(self.model.model.layers):
            modules = [("input_norm", layer.input_layernorm), ("qkv", layer.self_attn.qkv_proj),
                ("q_norm", layer.self_attn.q_norm), ("k_norm", layer.self_attn.k_norm),
                ("attention", layer.self_attn.attn), ("o_proj", layer.self_attn.o_proj),
                ("post_norm", layer.post_attention_layernorm), ("mlp", layer.mlp)]
            for name, module in modules:
                key = f"layer{index:02}/{name}"
                handles.append(module.register_forward_hook(
                    lambda m, args, output, key=key: store(key, output)))
            key = f"layer{index:02}/attention_inputs"
            handles.append(layer.self_attn.attn.register_forward_pre_hook(
                lambda m, args, key=key: store(key, args)))
        original_capture(self)

    ModelRunner.capture_cudagraph = traced_capture
    llm = None
    try:
        llm, report["initialization"] = create_engine(h1, pilot, config["variants"]["graph_no_prefix"])
        runner = llm.model_runner
        original_run = runner.run_model
        trace = []
        mode = {"dtype": None, "device": None}

        def traced_run(self, inputs, positions, is_prefill):
            old_dtype = torch.get_default_dtype()
            old_device = torch.get_default_device()
            try:
                if mode["dtype"] is not None:
                    torch.set_default_dtype(mode["dtype"])
                if mode["device"] is not None:
                    torch.set_default_device(mode["device"])
                output = original_run(inputs, positions, is_prefill)
            finally:
                torch.set_default_device(old_device)
                torch.set_default_dtype(old_dtype)
            if not is_prefill:
                trace.append({name: value.cpu().clone() for name, value in buffers.items()})
            return output

        runner.run_model = MethodType(traced_run, runner)
        case = cases["b1_p128_o64_none"]

        def run(graph, dtype=None, device=None):
            trace.clear()
            mode.update(dtype=dtype, device=device)
            runner.enforce_eager = not graph
            reset_cache(llm)
            runner.kv_cache.fill_(float("nan"))
            logits = {}
            for phase in ("cold", "warm"):
                _, logits[phase] = execute_wave(llm, case, phase, config, forced_tokens=4)
            return logits, list(trace)

        for label, static in (("native_tables", False), ("static_tables", True)):
            restore = install_shape_policy(runner, static_tables=static)
            baseline, baseline_trace = run(False)
            graph, graph_trace = run(True)
            dtype_eager, _ = run(False, dtype=torch.bfloat16)
            capture_defaults_eager, _ = run(False, dtype=torch.bfloat16, device="cuda")
            stages = []
            for step, (eager_values, graph_values) in enumerate(zip(baseline_trace, graph_trace)):
                for name, expected in eager_values.items():
                    actual = graph_values[name]
                    if not torch.equal(expected, actual):
                        stages.append({"decode_index": step, "stage": name,
                            "max_abs_error": (actual - expected).abs().max().item(),
                            "relative_rmse": ((actual-expected).square().mean()/expected.square().mean()).sqrt().item()})
            report["modes"][label] = {
                "eager_vs_graph": compare_logits(baseline, graph, pilot),
                "bf16_default_eager_vs_graph": compare_logits(dtype_eager, graph, pilot),
                "capture_defaults_eager_vs_graph": compare_logits(capture_defaults_eager, graph, pilot),
                "divergent_stages": stages}
            restore()
            print(label, "first divergent stages", stages[:6], flush=True)
        report["status"] = "pass"
        report["limitations"] = "Debug copies add kernels to captured forward; use this for localization, then validate any fix again without hooks."
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        ModelRunner.capture_cudagraph = original_capture
        try:
            if llm is not None:
                llm.model_runner.enforce_eager = False
                close_engine(llm)
        finally:
            report["finished_at_utc"] = now()
            write_result(report, args.result)
    for label, data in report["modes"].items():
        for name, rows in data.items():
            if name != "divergent_stages":
                print(label, name, max(x["max_abs_error"] for x in rows), all(x["pass"] for x in rows))


if __name__ == "__main__":
    main()

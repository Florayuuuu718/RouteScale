"""Test batch-1 Graph with compilation fallback forbidden in a fresh process."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MethodType
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_runtime import close_engine, create_engine, execute_wave, load_inputs, now, reset_cache
from serving.h2_numerical_policy import install_shape_policy
from serving.probe_h2_numerics import compare_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "results/h_nanovllm/h2_numerics/graph_compile_probe.json")
    args = parser.parse_args()
    if args.result.exists():
        raise SystemExit("Select a fresh result path")
    import torch
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    pilot, config, h1, cases = load_inputs()
    report = {"stage": "H2", "schema_version": 1, "status": "running",
        "evidence_class": "same_engine_numerical_diagnostic", "gpu_execution_allowed": False,
        "created_at_utc": now(), "entrypoint_sha256": digest(Path(__file__)),
        "recompile_limit": 64, "fail_on_recompile_limit_hit": True, "comparisons": {}}
    llm = None
    try:
        llm, report["initialization"] = create_engine(h1, pilot, config["variants"]["graph_no_prefix"])
        runner, case = llm.model_runner, cases["b1_p128_o64_none"]
        original = runner.run_model
        defaults = {"enabled": False}

        def scoped(self, inputs, positions, is_prefill):
            dtype, device = torch.get_default_dtype(), torch.get_default_device()
            try:
                if defaults["enabled"]:
                    torch.set_default_dtype(torch.bfloat16)
                    torch.set_default_device("cuda")
                return original(inputs, positions, is_prefill)
            finally:
                torch.set_default_device(device)
                torch.set_default_dtype(dtype)
        runner.run_model = MethodType(scoped, runner)

        def run(graph, capture_defaults=False):
            runner.enforce_eager = not graph
            defaults["enabled"] = capture_defaults
            reset_cache(llm)
            logits = {}
            for phase in ("cold", "warm"):
                _, logits[phase] = execute_wave(llm, case, phase, config, forced_tokens=4)
            return logits

        for label, static in (("native_tables", False), ("static_tables", True)):
            restore = install_shape_policy(runner, static_tables=static)
            eager, graph = run(False), run(True)
            aligned = run(False, True)
            report["comparisons"][label] = compare_logits(eager, graph, pilot)
            report["comparisons"][f"{label}_capture_defaults"] = compare_logits(aligned, graph, pilot)
            restore()
        report["status"] = "pass"
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm:
                llm.model_runner.enforce_eager = False
                close_engine(llm)
        finally:
            write_result(report, args.result)
    for label, rows in report["comparisons"].items():
        print(label, max(row["max_abs_error"] for row in rows), all(row["pass"] for row in rows))


if __name__ == "__main__":
    main()

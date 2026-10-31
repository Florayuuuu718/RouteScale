"""Same-engine causal interventions for the two H2 numerical failures, not timing."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_numerical_policy import install_shape_policy
from serving.h2_runtime import (
    DEFAULT_PILOT, close_engine, create_engine, execute_wave, load_inputs, now, provenance, reset_cache,
)


def compare_logits(reference, candidate, pilot):
    import torch
    rows = []
    for phase in ("cold", "warm"):
        for step, (expected, actual) in enumerate(zip(reference[phase], candidate[phase])):
            delta = actual - expected
            rmse = (delta.square().mean() / expected.square().mean()).sqrt().item()
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            close = torch.allclose(actual, expected, atol=pilot["correctness_atol"], rtol=pilot["correctness_rtol"])
            rows.append({"phase": phase, "step": step, "max_abs_error": delta.abs().max().item(),
                "relative_rmse": rmse, "exact": torch.equal(expected, actual),
                "top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).float().mean().item(),
                "pass": finite and close and rmse <= pilot["correctness_max_relative_rmse"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "results/h_nanovllm/h2_numerics/shape_probe.json")
    args = parser.parse_args()
    if args.result.exists():
        raise SystemExit("Preserve existing diagnostic evidence; select a new --result")
    pilot, config, h1, cases = load_inputs()
    report = {"schema_version": 1, "stage": "H2", "status": "running", "created_at_utc": now(),
        "evidence_class": "same_engine_numerical_diagnostic", "gpu_execution_allowed": False,
        "entrypoint_sha256": digest(Path(__file__)),
        "policy_sha256": digest(ROOT / "serving/h2_numerical_policy.py"), "comparisons": {}}
    llm = None
    try:
        import torch
        report["provenance"] = provenance(DEFAULT_PILOT, pilot, config, h1)
        llm, report["initialization"] = create_engine(h1, pilot, config["variants"]["graph_no_prefix"])
        runner = llm.model_runner

        def run(case, *, graph=False, prefix=False):
            runner.enforce_eager = not graph
            llm._h2_prefix_cache = prefix
            reset_cache(llm)
            runner.kv_cache.fill_(float("nan"))
            result = {}
            for phase in ("cold", "warm"):
                _, result[phase] = execute_wave(llm, case, phase, config, forced_tokens=4)
            return result

        graph_case = cases["b1_p128_o64_none"]
        reference = run(graph_case)
        repeated = run(graph_case)
        graph = run(graph_case, graph=True)
        report["comparisons"]["batch1_eager_aa"] = compare_logits(reference, repeated, pilot)
        report["comparisons"]["batch1_native_eager_vs_graph"] = compare_logits(reference, graph, pilot)
        restore = install_shape_policy(runner, static_tables=True)
        padded = run(graph_case)
        padded_graph = run(graph_case, graph=True)
        report["comparisons"]["batch1_static_tables_eager_vs_graph"] = compare_logits(padded, padded_graph, pilot)
        report["comparisons"]["batch1_native_vs_padded_eager"] = compare_logits(reference, padded, pilot)
        restore()

        prefix_case = cases["b8_p512_o64_high"]
        for label, static, paged in (("native", False, False), ("static_tables", True, False),
                                     ("paged_prefill", False, True), ("static_paged", True, True)):
            restore = install_shape_policy(runner, static_tables=static, paged_prefill=paged)
            reference = run(prefix_case)
            candidate = run(prefix_case, prefix=True)
            report["comparisons"][f"prefix_{label}"] = compare_logits(reference, candidate, pilot)
            restore()
            rows = report["comparisons"][f"prefix_{label}"]
            print(label, "max error", max(r["max_abs_error"] for r in rows), "pass", all(r["pass"] for r in rows), flush=True)
        report["status"] = "pass"
        report["limitations"] = "Diagnostic completed, not a formal timing gate. Historical native results remain immutable."
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                llm.model_runner.enforce_eager = False  # this engine owns captured graphs
                close_engine(llm)
        finally:
            report["finished_at_utc"] = now()
            write_result(report, args.result)
    for name, rows in report["comparisons"].items():
        print(name, "max_error", max(r["max_abs_error"] for r in rows), "pass", all(r["pass"] for r in rows))


if __name__ == "__main__":
    main()

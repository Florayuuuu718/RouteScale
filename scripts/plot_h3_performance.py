"""Render measured H3 throughput with independent-process min/max ranges.

Run in an isolated reporting environment; do not change the serving packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/h_nanovllm/h3_performance_v5"))
    args = parser.parse_args()
    summary_path = args.directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "pass" or summary["formal_processes"] != 36:
        raise ValueError("A completed, audited matrix is required")
    if not (args.directory / "audit.json").is_file():
        raise ValueError("Independent audit is required before plotting")
    cases = ["latency_short", "batch_short", "page_growth", "long_prefill"]
    variants = ["native_kv", "paged_eager", "paged_graph"]
    rows = {(r["measurement_scope"], r["case_id"], r["variant"]): r for r in summary["variants"]}
    colours = ["#677483", "#3577B5", "#16856D"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
        for axis, scope, title in zip(axes, ("model_execution", "replay_e2e"),
                ("Model execution · CUDA Events", "Fixed-token replay · wall time")):
            x = np.arange(len(cases))
            for offset, variant, colour in zip((-.26, 0, .26), variants, colours):
                selected = [rows[scope, case, variant] for case in cases]
                values = np.array([r["output_tokens_per_second"] for r in selected])
                lower = values - np.array([min(r["throughput_run_values"]) for r in selected])
                upper = np.array([max(r["throughput_run_values"]) for r in selected]) - values
                bars = axis.bar(x + offset, values, .24, color=colour, label=variant,
                    yerr=np.stack((lower, upper)), capsize=3, error_kw={"linewidth": 1})
                axis.bar_label(bars, fmt="%.0f", fontsize=8, padding=4)
            axis.set_xticks(x, cases, rotation=18, ha="right")
            axis.set_ylabel("Output tokens / second")
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.set_title(title, loc="left", fontweight="bold")
            axis.set_ylim(top=max(r["output_tokens_per_second"] for r in summary["variants"] if r["measurement_scope"] == scope) * 1.7)
            axis.grid(axis="y", alpha=.18)
            axis.set_axisbelow(True)
        axes[0].legend(frameon=False, fontsize=9, loc="upper left")
        fig.suptitle("RouteScale Dense · same BF16 weights · GEMM reduced-precision reduction off\n"
            "Median of 3 independent processes; whiskers show min–max. Fixed replay, not online serving.",
            fontsize=12)
        for extension in ("svg", "png"):
            fig.savefig(args.directory / f"throughput.{extension}", dpi=180)
        plt.close(fig)
    metadata = dict(matplotlib=matplotlib.__version__, numpy=np.__version__,
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest()
            for name in ("throughput.svg", "throughput.png")})
    (args.directory / "plot_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("H3 measured throughput SVG/PNG: saved")


if __name__ == "__main__":
    main()

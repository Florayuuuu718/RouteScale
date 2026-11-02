"""Read-only metric recomputation for the bounded H3 fixed-token experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.h3_performance import SCOPES, VARIANTS, value_hash


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Empty timing sample")
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    return ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)


def summarize_traces(traces, scope):
    if not traces:
        raise ValueError("Missing wave traces")
    wall = [t["wall_ms"] for t in traces]
    denom = [t["model_execution_ms"] if scope == "model_execution" else t["wall_ms"] for t in traces]
    itl = [v for t in traces for v in t["itl_ms"]]
    return dict(median_ttft_ms=median(t["ttft_ms"] for t in traces),
        median_tpot_ms=median(t["tpot_ms"] for t in traces),
        median_wall_ms=median(wall),
        median_model_execution_ms=median(denom) if scope == "model_execution" else None,
        output_tokens_per_second=median(t["output_tokens"] * 1000 / d for t, d in zip(traces, denom)),
        itl_p50_ms=quantile(itl, .5), itl_p95_ms=quantile(itl, .95), itl_p99_ms=quantile(itl, .99),
        wall_cv=pstdev(wall) / (sum(wall) / len(wall)),
        measurement_cv=pstdev(denom) / (sum(denom) / len(denom)))


def validate_run(run, config, case, provenance, gate_hash, *, pilot=False):
    if (run.get("status") != "pass" or run.get("stage") != "H3"
            or run.get("benchmark") != "routescale_performance"
            or run.get("evidence_class") != ("pilot_gpu" if pilot else "formal_gpu")
            or run.get("variant") not in VARIANTS or run.get("case_id") != case["case_id"]):
        raise ValueError("Failed, CPU, wrong-scope or unknown process")
    if (run["provenance"] != provenance or run["qualification_gate_sha256"] != gate_hash
            or run["workload_sha256"] != case["workload_sha256"] or run["dtype"] != "bfloat16"
            or run["prefix_cache"] is not False or run["token_policy"] != config["token_policy"]):
        raise ValueError("Provenance, workload, precision or gate mismatch")
    expected = config["pilot_measure_repeats" if pilot else "measure_repeats"]
    if (run["measure_repeats"] != expected
            or run["warmup_repeats"] != config["pilot_warmup_repeats" if pilot else "warmup_repeats"]
            or run["free_bytes_after_run"] < config["minimum_free_bytes"]
            or set(run["measurements"]) != set(SCOPES)):
        raise ValueError("Timing policy or memory budget changed")
    expected_kv = case["batch_size"] * 8 * 1024 * 1024 if run["variant"] == "native_kv" else 64 * 1024 * 1024
    if run["kv_bytes"] != expected_kv:
        raise ValueError("Unexpected actual KV allocation")
    for scope, measurement in run["measurements"].items():
        traces = measurement["traces"]
        if len(traces) != expected or measurement["peak_reserved_bytes"] < measurement["peak_allocated_bytes"]:
            raise ValueError("Repeat count or memory metrics invalid")
        if scope == "replay_e2e" and len({t["argmax_sha256"] for t in traces}) != 1:
            raise ValueError("Same-path repeated argmax outputs changed")
        for trace in traces:
            values = [trace["ttft_ms"], trace["tpot_ms"], trace["wall_ms"], *trace["itl_ms"]]
            if scope == "model_execution":
                values.append(trace["model_execution_ms"])
                if not math.isclose(trace["model_execution_ms"], trace["ttft_ms"] + sum(trace["itl_ms"]), rel_tol=1e-12):
                    raise ValueError("Execution interval does not match per-step events")
            elif trace["model_execution_ms"] is not None or trace["wall_ms"] < trace["ttft_ms"] + sum(trace["itl_ms"]):
                raise ValueError("Invalid wall-time token visibility")
            if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Nonpositive or nonfinite timings")
            if (len(trace["itl_ms"]) != case["output_length"] - 1
                    or not math.isclose(trace["tpot_ms"], sum(trace["itl_ms"]) / len(trace["itl_ms"]), rel_tol=1e-12)
                    or trace["output_tokens"] != case["batch_size"] * case["output_length"]
                    or trace["output_sha256"] != value_hash(case["continuation_ids"])
                    or trace["cached_positions"] != case["prompt_length"] + case["output_length"] - 1
                    or trace["prefix_tokens"] != 0 or not trace["released"]
                    or trace["prefill_calls"] != 1 or trace["decode_calls"] != case["output_length"] - 1):
                raise ValueError("Output budget, tokens, cache or lifecycle contract changed")
            blocks = 0 if run["variant"] == "native_kv" else case["batch_size"] * math.ceil(trace["cached_positions"] / 256)
            if trace["max_used_blocks"] != blocks:
                raise ValueError("Incorrect real page growth")


def summarize_runs(path_runs, config, cases, provenance, gate_hash):
    by_case = {case["case_id"]: case for case in cases}
    planned = {(case, variant, f"run{run}") for case in by_case for variant in VARIANTS
        for run in range(1, config["independent_runs"] + 1)}
    observed = [(r["case_id"], r["variant"], r["run_id"]) for _, r in path_runs]
    if len(observed) != len(planned) or set(observed) != planned:
        raise ValueError("Missing, duplicated or unexpected formal process")
    groups = {}
    for path, run in path_runs:
        validate_run(run, config, by_case[run["case_id"]], provenance, gate_hash)
        groups.setdefault((run["case_id"], run["variant"]), []).append((path, run))
    variants = []
    for (case_id, variant), members in sorted(groups.items()):
        members.sort(key=lambda row: row[1]["run_id"])
        for scope in SCOPES:
            stats = [summarize_traces(run["measurements"][scope]["traces"], scope) for _, run in members]
            metrics = {key: median(s[key] for s in stats) if stats[0][key] is not None else None for key in stats[0]}
            throughput = [s["output_tokens_per_second"] for s in stats]
            variants.append(dict(case_id=case_id, variant=variant, measurement_scope=scope,
                **metrics, independent_runs=config["independent_runs"],
                throughput_run_values=throughput, throughput_max_min_ratio=max(throughput) / min(throughput),
                median_peak_allocated_bytes=median(r["measurements"][scope]["peak_allocated_bytes"] for _, r in members),
                median_peak_reserved_bytes=median(r["measurements"][scope]["peak_reserved_bytes"] for _, r in members),
                median_initialization_seconds=median(r["initialization"]["total_seconds"] for _, r in members),
                median_graph_capture_seconds=median(r["initialization"]["graph_capture_seconds"] for _, r in members),
                actual_kv_bytes=members[0][1]["kv_bytes"], raw_files=[str(p.relative_to(ROOT)) for p, _ in members]))
    lookup = {(r["case_id"], r["variant"], r["measurement_scope"]): r for r in variants}
    comparisons = []
    for case_id in by_case:
        for scope in SCOPES:
            for base, changed in (("native_kv", "paged_eager"), ("paged_eager", "paged_graph"), ("native_kv", "paged_graph")):
                a, b = lookup[case_id, base, scope], lookup[case_id, changed, scope]
                comparisons.append(dict(case_id=case_id, measurement_scope=scope, baseline=base, candidate=changed,
                    output_throughput_ratio=b["output_tokens_per_second"] / a["output_tokens_per_second"],
                    ttft_speedup=a["median_ttft_ms"] / b["median_ttft_ms"],
                    tpot_speedup=a["median_tpot_ms"] / b["median_tpot_ms"],
                    extra_peak_allocated_bytes=b["median_peak_allocated_bytes"] - a["median_peak_allocated_bytes"]))
    return dict(schema_version=1, stage="H3", status="pass", scope=config["scope"],
        formal_processes=len(path_runs), independent_runs=config["independent_runs"],
        provenance=provenance, qualification_gate_sha256=gate_hash, variants=variants, comparisons=comparisons,
        limitations="Controlled offline fixed-token replay; native/paged changes kernels, layout and control. "
        "Not public generate/text/network timing, free-generation quality, online SLO or maximum capacity.")


def markdown_table(summary):
    lines = ["# H3 同模型性能结果", "", "三次独立进程的中位数；固定 token 回放，全部 BF16，Prefix off。", "",
        "| 范围 | case | 路径 | 输出 token/s | TTFT ms | TPOT ms | 峰值 allocated MiB |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in summary["variants"]:
        lines.append(f"| {row['measurement_scope']} | {row['case_id']} | {row['variant']} | "
            f"{row['output_tokens_per_second']:.1f} | {row['median_ttft_ms']:.3f} | "
            f"{row['median_tpot_ms']:.3f} | {row['median_peak_allocated_bytes']/2**20:.1f} |")
    return "\n".join(lines) + "\n"

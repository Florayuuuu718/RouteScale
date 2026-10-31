"""Validate the H2 raw-result contract and aggregate single-variable comparisons."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import write_result


VARIANTS = {
    "eager_no_prefix": {"enforce_eager": True, "prefix_cache": False},
    "graph_no_prefix": {"enforce_eager": False, "prefix_cache": False},
    "eager_prefix": {"enforce_eager": True, "prefix_cache": True},
}
LATENCIES = [f"{metric}_{quantile}_ms" for metric in ("ttft", "tpot", "e2e")
             for quantile in ("p50", "p95", "p99")]
METRICS = LATENCIES + ["output_tokens_per_second", "peak_allocated_bytes"]


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_run(run):
    if (run.get("schema_version"), run.get("stage"), run.get("status"), run.get("evidence_class")) != (1, "H2", "pass", "formal_gpu"):
        raise ValueError("H2 summary accepts passed formal GPU results only")
    variant, engine, workload = run["variant"], run["engine"], run["workload"]
    if engine.get("dtype") != "bfloat16":
        raise ValueError("H2 requires the pinned BF16 precision")
    if any(type(engine.get(k)) is not bool for k in ("enforce_eager", "prefix_cache")):
        raise ValueError("Engine switches must be booleans")
    if variant not in VARIANTS or any(engine.get(k) != v for k, v in VARIANTS[variant].items()):
        raise ValueError("Variant label does not match engine switches")
    for field in ("upstream_commit", "model_revision", "model_sha256", "tokenizer_sha256", "source_sha256", "environment"):
        if not run["provenance"].get(field):
            raise ValueError(f"Missing provenance: {field}")
    if not workload.get("sha256") or workload.get("cache_reset_policy") != "fresh_block_manager_per_pair":
        raise ValueError("Missing workload hash or invalid cache reset policy")
    if not workload.get("sampling", {}).get("ignore_eos"):
        raise ValueError("H2 requires a fixed output-token budget with ignore_eos")
    batch, prompt, output, pairs = [workload[k] for k in
        ("batch_size", "prompt_length", "output_length", "measured_pairs")]
    if any(type(x) is not int or x <= 0 for x in (batch, prompt, output, pairs)):
        raise ValueError("Invalid workload dimensions")
    if engine.get("max_num_batched_tokens", 0) < batch * prompt:
        raise ValueError("Cold wave requires bulk admission before prefix publication")
    if engine.get("max_num_seqs", 0) < batch or engine.get("max_model_len", 0) < prompt + output:
        raise ValueError("Engine capacity is below the requested workload")
    if type(engine.get("num_kvcache_blocks")) is not int or engine["num_kvcache_blocks"] <= 0:
        raise ValueError("Actual KV pool size must be reported")
    block = engine.get("kvcache_block_size")
    if type(block) is not int or block <= 0 or block % 256:
        raise ValueError("Invalid upstream KV block size")
    if engine["num_kvcache_blocks"] < batch * ((prompt + output - 2) // block + 1):
        raise ValueError("KV pool cannot cover the cold-wave decode budget without preemption")
    for phase in ("cold", "warm"):
        metrics = run["phases"][phase]
        for key in METRICS:
            value = metrics[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid metric: {phase}.{key}")
        for name in ("ttft", "tpot", "e2e"):
            if not metrics[f"{name}_p50_ms"] <= metrics[f"{name}_p95_ms"] <= metrics[f"{name}_p99_ms"]:
                raise ValueError("Latency quantiles are out of order")
        expected = {"requests_completed": batch * pairs, "prompt_tokens": batch * pairs * prompt,
                    "output_tokens": batch * pairs * output}
        if any(metrics.get(key) != value for key, value in expected.items()):
            raise ValueError("Incomplete requests or unequal measured token budget")
        hits = metrics.get("cached_prompt_tokens")
        if type(hits) is not int or not 0 <= hits <= expected["prompt_tokens"]:
            raise ValueError("Invalid cached-token count")
        if hits and (phase == "cold" or not engine["prefix_cache"] or prompt <= engine["kvcache_block_size"]):
            raise ValueError("Unexpected cache hits in a cold/disabled/short-prompt control")


def summarize_runs(path_runs, expected_runs=3, *, mechanism="both"):
    if not path_runs or expected_runs < 3:
        raise ValueError("At least three independent runs are required")
    if mechanism not in ("both", "cuda_graph"):
        raise ValueError("Unknown summary scope")
    expected_variants = set(VARIANTS) if mechanism == "both" else {"eager_no_prefix", "graph_no_prefix"}
    groups = {}
    for path, run in path_runs:
        validate_run(run)
        groups.setdefault((run["case_id"], run["variant"]), []).append((path, run))
    variants, by_case = [], {}
    for (case, name), members in sorted(groups.items()):
        members.sort(key=lambda pair: pair[1]["run_id"])
        if [r["run_id"] for _, r in members] != [f"run{i}" for i in range(1, expected_runs + 1)]:
            raise ValueError(f"Incomplete or duplicated run IDs: {case}/{name}")
        runs = [r for _, r in members]
        invariants = [{k: r[k] for k in ("provenance", "workload", "engine")} for r in runs]
        if len({stable(value) for value in invariants}) != 1:
            raise ValueError(f"Independent-run invariants differ: {case}/{name}")
        row = {"case_id": case, "variant": name, **invariants[0],
            "run_files": [str(p) for p, _ in members], "phases": {}}
        for phase in ("cold", "warm"):
            row["phases"][phase] = {key: median(r["phases"][phase][key] for r in runs) for key in
                METRICS + ["requests_completed", "prompt_tokens", "output_tokens", "cached_prompt_tokens"]}
            row["phases"][phase]["prompt_token_hit_rate"] = (
                row["phases"][phase]["cached_prompt_tokens"] / row["phases"][phase]["prompt_tokens"])
        variants.append(row)
        by_case.setdefault(case, {})[name] = row
    comparisons = []
    for case, sides in sorted(by_case.items()):
        if set(sides) != expected_variants:
            raise ValueError(f"Missing single-variable control: {case}")
        baseline = sides["eager_no_prefix"]
        def controls(row):
            return {"provenance": row["provenance"], "workload": row["workload"],
                "engine": {k: v for k, v in row["engine"].items() if k not in ("enforce_eager", "prefix_cache")}}
        if len({stable(controls(row)) for row in sides.values()}) != 1:
            raise ValueError(f"Cross-variant invariants differ: {case}")
        for name, mechanism in (("graph_no_prefix", "cuda_graph"), ("eager_prefix", "prefix_cache")):
            if name not in expected_variants:
                continue
            for phase in ("cold", "warm"):
                base, changed = baseline["phases"][phase], sides[name]["phases"][phase]
                comparisons.append({"case_id": case, "mechanism": mechanism, "phase": phase,
                    "output_throughput_ratio": changed["output_tokens_per_second"] / base["output_tokens_per_second"],
                    "ttft_speedup": base["ttft_p50_ms"] / changed["ttft_p50_ms"],
                    "e2e_speedup": base["e2e_p50_ms"] / changed["e2e_p50_ms"],
                    "extra_peak_allocated_bytes": changed["peak_allocated_bytes"] - base["peak_allocated_bytes"],
                    "prompt_token_hit_rate": changed["prompt_token_hit_rate"]})
    return {"schema_version": 1, "stage": "H2", "status": "pass", "benchmark": "nanovllm_ablation_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "independent_runs": expected_runs,
        "scope": "cuda_graph_only" if expected_variants != set(VARIANTS) else "cuda_graph_and_prefix_cache",
        "aggregation": "median of independent run statistics, not pooled tail quantiles",
        "variant_count": len(variants), "comparisons": comparisons, "variants": variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results/h_nanovllm/h2_formal/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "results/h_nanovllm/h2_formal/summary.json")
    parser.add_argument("--mechanism", choices=("both", "cuda_graph"), default="both")
    args = parser.parse_args()
    runs = [(p, json.loads(p.read_text())) for p in sorted(args.raw_dir.rglob("run*.json"))]
    result = summarize_runs(runs, mechanism=args.mechanism)
    write_result(result, args.output)
    print(f"H2 summary: {result['variant_count']} variants -> {args.output}")


if __name__ == "__main__":
    main()

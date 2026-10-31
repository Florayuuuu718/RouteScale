"""CPU-only schedules, accounting and fail-closed diagnostic report validation."""

from __future__ import annotations

import math
import statistics

from serving.h2_kv_pressure import validate_pressure_trace

LAYOUTS = ("stride4", "dense0", "dense8")
POLICIES = ("static_window", "paged")


def paired_order(run, block):
    if run not in (1,2,3) or block not in range(4):
        raise ValueError("Outside bounded paired diagnostic")
    a,b = POLICIES if (run+block)%2 else tuple(reversed(POLICIES))
    return (a,b,b,a)


def layout_order(run, round_id):
    if run not in (1,2,3) or round_id not in range(3):
        raise ValueError("Outside bounded layout diagnostic")
    offset = (run-1+round_id)%3
    return LAYOUTS[offset:]+LAYOUTS[:offset]


def layout_pages(layout):
    if layout not in LAYOUTS:
        raise ValueError("Unknown controlled page layout")
    return list(range(0,32,4)) if layout == "stride4" else list(range(8 if layout == "dense8" else 0,16 if layout == "dense8" else 8))


def wall_components(trace):
    prefill,decode = 0.0,0.0
    for step in trace["steps"]:
        elapsed = step["token_visible_ms"]-step["model_start_ms"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid model-visible interval")
        if step["is_prefill"]:
            prefill += elapsed
        else:
            decode += elapsed
    outside = trace["wall_seconds"]*1000-prefill-decode
    if outside < -1e-6:
        raise ValueError("Intervals exceed trace wall time")
    return dict(prefill_visible_ms=prefill,decode_visible_ms=decode,outside_run_ms=outside)


def layout_metrics(trace):
    if (trace["evidence_class"] != "teacher_forced_cuda_stream_probe" or trace["layout"] not in LAYOUTS
            or trace["pages"] != layout_pages(trace["layout"]) or len(trace["steps"]) != 64
            or not trace["released"] or trace["sampler_executed"]):
        raise ValueError("Layout probe scope/coverage changed; not real serving throughput")
    for i,step in enumerate(trace["steps"]):
        if step["is_prefill"] != (i==0) or step["cached_tokens"] != 128+i:
            raise ValueError("Layout token/cache position changed")
        if any(not math.isfinite(step[k]) or step[k] <= 0 for k in ("stream_ms","host_model_ms")):
            raise ValueError("Invalid CUDA-event or synchronized host timing")
    decode = trace["steps"][1:]
    return dict(prefill_stream_ms=trace["steps"][0]["stream_ms"],
        decode_stream_mean_ms=statistics.mean(s["stream_ms"] for s in decode),
        decode_host_mean_ms=statistics.mean(s["host_model_ms"] for s in decode))


def validate_report(report,config,case,pressure,provenance):
    if (report["status"] != "pass" or report["evidence_class"] != "gpu_kv_negative_control_diagnostic"
            or report["provenance"] != provenance or report["run_id"] not in (1,2,3)
            or report["kv_tensor_bytes"] != 939524096 or len(report["paired_blocks"]) != 4
            or set(report["compile_counters"]) != {"paired","layouts"}
            or report["compiler_settings"] != dict(recompile_limit=64,fail_on_recompile_limit_hit=True,suppress_errors=False)
            or report["free_bytes_after_run"] < 536870912):
        raise ValueError("Diagnostic source/engine/settings or bounded scope changed")
    for counts in report["compile_counters"].values():
        if counts["before"] != counts["after"]:
            raise ValueError("Compilation inside a diagnostic measurement")
    for block_id,block in enumerate(report["paired_blocks"]):
        order = paired_order(report["run_id"],block_id)
        if block["block_id"] != block_id or [r["policy"] for r in block["traces"]] != list(order):
            raise ValueError("Paired order lost balance")
        hashes=[]
        for row in block["traces"]:
            if row["seed"] != report["seed"]+block_id or row["trace"]["finite_logits_checked"]:
                raise ValueError("Sampled paired trace seed/probe contract changed")
            validate_pressure_trace(row["trace"],case,pressure,row["policy"])
            if row["components"] != wall_components(row["trace"]):
                raise ValueError("Wall phase accounting changed")
            hashes.append([r["output_token_ids_sha256"] for r in row["trace"]["requests"]])
        if any(row != hashes[0] for row in hashes):
            raise ValueError("Same-shape paired sampled outputs changed")
    expected=[(i,layout) for i in range(3) for layout in layout_order(report["run_id"],i)]
    if [(r["round_id"],r["trace"]["layout"]) for r in report["layout_timings"]] != expected:
        raise ValueError("Incomplete/unbalanced layout timing schedule")
    for row in report["layout_timings"]:
        if row["metrics"] != layout_metrics(row["trace"]):
            raise ValueError("Saved CUDA-stream metrics did not recompute")
    if set(report["layout_numerics"]) != set(LAYOUTS):
        raise ValueError("Missing layout numerical gate")
    for layout,n in report["layout_numerics"].items():
        if (n["vectors"] != 512 or not n["finite"] or not n["kv_poisoned"] or not n["released"]
                or not n["free_pages_remain_nan"]
                or n["pages"] != layout_pages(layout) or n["shape"] != [64,8,151936]
                or n["dtype"] != "bfloat16" or (layout != "stride4" and n["exact_vectors"] != 512)):
            raise ValueError("Layout timing requires its own exact poisoned-KV gate")


def summarize(reports,config,case,pressure,provenance):
    if sorted(r["run_id"] for r in reports) != [1,2,3]:
        raise ValueError("Need exactly three independent diagnostic processes")
    blocks,per_run=[],[]
    for report in sorted(reports,key=lambda r:r["run_id"]):
        validate_report(report,config,case,pressure,provenance)
        ratios=[]
        for block in report["paired_blocks"]:
            durations={p:sum(row["trace"]["wall_seconds"] for row in block["traces"] if row["policy"]==p) for p in POLICIES}
            ratio=durations["static_window"]/durations["paged"]
            ratios.append(ratio)
            blocks.append(dict(run_id=report["run_id"],block_id=block["block_id"],paged_throughput_ratio=ratio))
        layout_rows={}
        for layout in LAYOUTS:
            rows=[r["metrics"] for r in report["layout_timings"] if r["trace"]["layout"]==layout]
            layout_rows[layout]={key:statistics.median(r[key] for r in rows) for key in rows[0]}
        per_run.append(dict(run_id=report["run_id"],paired_ratio_median=statistics.median(ratios),
            paired_ratios=ratios,layouts=layout_rows))
    return dict(status="pass",evidence_class="negative_control_diagnostic_summary",independent_runs=3,
        paired_sampled_traces=48,layout_stream_probes=27,full_vocabulary_layout_comparisons=3072,
        paired_blocks=blocks,per_run=per_run,
        median_of_process_paired_ratios=statistics.median(r["paired_ratio_median"] for r in per_run),
        layouts={layout:{key:statistics.median(r["layouts"][layout][key] for r in per_run)
            for key in per_run[0]["layouts"][layout]} for layout in LAYOUTS},
        limitations=config["limitations"])

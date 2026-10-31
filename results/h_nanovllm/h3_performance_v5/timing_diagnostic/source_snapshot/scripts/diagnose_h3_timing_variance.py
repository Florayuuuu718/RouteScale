"""Three bounded, balanced long-Prefill controls; never replace formal data."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_performance import DEFAULT_CONFIG, VARIANTS, load_inputs, create_path, run_wave, now
from scripts.summarize_h3_performance import summarize_traces


def sensors():
    result = subprocess.run(["nvidia-smi", "--query-gpu=pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return dict(gpu=result.stdout.strip(), cpu_load=list(os.getloadavg()))


def worker(config_path, run_id):
    import torch
    config, artifact, cases, provenance = load_inputs(config_path)
    directory = ROOT / config["results_dir"] / "timing_diagnostic"
    output = directory / f"run{run_id}.json"
    if output.exists():
        raise ValueError("Preserve existing paired diagnostic")
    case = next(c for c in cases if c["case_id"] == "long_prefill")
    order = VARIANTS[(run_id - 1) % 3:] + VARIANTS[:(run_id - 1) % 3]
    report = dict(schema_version=1, stage="H3", status="running", evidence_class="paired_timing_diagnostic",
        created_at_utc=now(), source_sha256=digest(Path(__file__)), provenance=provenance, order=list(order), variants={})
    for variant in order:
        session = create_path(artifact, case, config, variant)
        try:
            for _ in range(config["warmup_repeats"]):
                run_wave(session, case, "replay_e2e")
            before = sensors()
            traces = [run_wave(session, case, "replay_e2e")[0] for _ in range(config["measure_repeats"])]
            if any(t["output_sha256"] != traces[0]["output_sha256"] or not t["released"] for t in traces):
                raise ValueError("Paired replay or lifecycle changed")
            report["variants"][variant] = dict(traces=traces, statistics=summarize_traces(traces, "replay_e2e"),
                sensors_before=before, sensors_after=sensors(), weight_sha256=session.weight_sha256)
        finally:
            session.close()
        gc.collect()
        torch.cuda.empty_cache()
    report.update(status="pass", limitations="Three fresh processes; within-process path rotation, E2E only. "
        "No power/clock setting change; snapshots do not establish a unique historical cause. Formal 36-process data unchanged.")
    write_result(report, output)
    print(json.dumps({v: round(r["statistics"]["output_tokens_per_second"], 2) for v, r in report["variants"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", type=int)
    args = parser.parse_args()
    if args.run_id is not None:
        worker(args.config, args.run_id)
        return
    config = json.loads(args.config.read_text())
    directory = ROOT / config["results_dir"] / "timing_diagnostic"
    if directory.exists():
        raise ValueError("Preserve existing timing diagnosis")
    directory.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4", TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/h3_timing_diagnostic/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "runs/cache/h3_timing_diagnostic/triton"))
    for run_id in (1, 2, 3):
        print(f"H3 timing control {run_id}/3", flush=True)
        with (directory / f"run{run_id}.log").open("x") as log:
            subprocess.run([str(ROOT / "runs/envs/nano-vllm/bin/python"), str(Path(__file__).resolve()),
                "--config", str(args.config), "--run-id", str(run_id)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    reports = [json.loads((directory / f"run{i}.json").read_text()) for i in (1, 2, 3)]
    ratios = {baseline: [r["variants"]["paged_graph"]["statistics"]["output_tokens_per_second"]
        / r["variants"][baseline]["statistics"]["output_tokens_per_second"] for r in reports]
        for baseline in ("native_kv", "paged_eager")}
    write_result(dict(schema_version=1, stage="H3", status="pass", evidence_class="paired_timing_diagnostic_summary",
        created_at_utc=now(), paired_throughput_ratios=ratios,
        raw_sha256={f"run{i}.json": digest(directory / f"run{i}.json") for i in (1, 2, 3)},
        limitations="Additional long-Prefill diagnostic, not replacements for formal medians or a unique slowdown diagnosis."),
        directory / "summary.json")
    print(json.dumps(ratios, indent=2), flush=True)


if __name__ == "__main__":
    main()

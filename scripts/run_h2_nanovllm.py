"""Run only the bounded H2 pilot, sequentially on one GPU, behind a numerical gate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h2_runtime import DEFAULT_PILOT, SOURCE_FILES, now


def pilot_jobs(config, variants=None):
    # Rotate variant order across independent runs; do not put all Graph runs at
    # the end of the laptop's thermal/power history.
    names = list(variants or config["variants"])
    for index in range(config["independent_runs"]):
        for case_id in config["pilot_case_ids"]:
            shift = index % len(names)
            for name in names[shift:] + names[:shift]:
                yield case_id, name, f"run{index + 1}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if "," in args.gpu:
        raise ValueError("Expose one GPU only")
    pilot = json.loads(args.config.read_text())
    config = json.loads((ROOT / pilot["h2_config"]).read_text())
    directory = ROOT / pilot["results_dir"]
    journal_path = directory / "run.json"
    lock_path = ROOT / "runs/tmp/h2_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An H2 runner is already active")
    if journal_path.exists() and not args.resume:
        raise SystemExit("H2 pilot already has a journal; use --resume after inspecting it")
    python = str(ROOT / "runs/envs/nano-vllm/bin/python")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/triton"), OMP_NUM_THREADS="4")
    names = list(pilot.get("variants", config["variants"]))
    job_config = {**config, "pilot_case_ids": pilot.get("pilot_case_ids", config["pilot_case_ids"])}
    if not set(job_config["pilot_case_ids"]) <= set(config["pilot_case_ids"]):
        raise ValueError("GPU scope cannot expand the fixed pilot case list")
    logs = ROOT / "runs/logs/h_nanovllm" / directory.name
    logs.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "stage": "H2", "status": "running", "pid": os.getpid(),
        "started_at_utc": now(), "completed_steps": [], "planned_processes": len(list(pilot_jobs(job_config, names))),
        "case_ids": job_config["pilot_case_ids"], "variants": names,
        "scope": pilot.get("scope", "cuda_graph_and_prefix_cache"),
        "pilot_config_sha256": digest(args.config), "completed_formal_processes": 0}

    def run(name, command):
        report.update(current_step=name, updated_at_utc=now())
        write_result(report, journal_path)
        print(f"H2: {name}", flush=True)
        path = logs / f"{name.replace('/', '_')}.log"
        with path.open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=900, check=True)
        report["completed_steps"].append(name)
        write_result(report, journal_path)

    try:
        gate_path = directory / "preflight.json"
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
        valid_gate = (args.resume and gate and gate.get("status") == "pass"
            and gate["provenance"]["pilot_config_sha256"] == digest(args.config)
            and gate["provenance"]["source_sha256"] == {name: digest(ROOT / name) for name in SOURCE_FILES})
        if not valid_gate:
            for case_id in pilot.get("correctness_case_ids", [pilot["correctness_case_id"]]):
                for name in names:
                    run(f"preflight/{case_id}/{name}", [python, "-m", "serving.benchmark_nanovllm",
                        "--config", str(args.config), "--case-id", case_id,
                        "--variant", name, "--preflight", "--result", str(directory / "preflight" / case_id / f"{name}.json")])
            run("preflight/compare", [python, "-m", "serving.verify_h2_preflight", "--config", str(args.config)])
        else:
            report["completed_steps"].append("preflight/reused_verified_source_gate")
        if not args.preflight_only:
            for case_id, name, run_id in pilot_jobs(job_config, names):
                path = directory / "raw" / case_id / name / f"{run_id}.json"
                if args.resume and path.exists():
                    old = json.loads(path.read_text())
                    if old.get("status") == "pass" and old.get("preflight_sha256") == digest(gate_path):
                        from scripts.summarize_h2_nanovllm import validate_run
                        validate_run(old)
                        report["completed_formal_processes"] += 1
                        continue
                    if old.get("status") == "pass":
                        raise RuntimeError(f"Passed result has a stale gate; preserve it and use a new results_dir: {path}")
                run(f"{case_id}/{name}/{run_id}", [python, "-m", "serving.benchmark_nanovllm",
                    "--config", str(args.config), "--case-id", case_id, "--variant", name,
                    "--run-id", run_id, "--result", str(path)])
                report["completed_formal_processes"] += 1
            command = [python, "scripts/summarize_h2_nanovllm.py", "--raw-dir", str(directory / "raw"),
                "--output", str(directory / "summary.json")]
            if pilot.get("scope") == "cuda_graph_only":
                command += ["--mechanism", "cuda_graph"]
            run("summarize", command)
        report.update(status="pass", current_step="preflight_complete" if args.preflight_only else "complete")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"] = now()
        write_result(report, journal_path)
    print(f"H2 pilot: {report['status']} -> {journal_path}", flush=True)


if __name__ == "__main__":
    main()

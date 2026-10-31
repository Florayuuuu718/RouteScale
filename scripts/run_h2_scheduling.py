"""Numerically qualify admission policies, then run a bounded 18-process pilot."""

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
from serving.h2_runtime import now
from serving.h2_scheduling import DEFAULT_SCHEDULING, POLICIES, SCHEDULING_SOURCES, load_scheduling_inputs
from scripts.summarize_h2_scheduling import validate_run


def scheduling_jobs(config):
    for index in range(config["independent_runs"]):
        for case in config["cases"]:
            names = list(POLICIES)
            if index % 2:
                names.reverse()
            for policy in names:
                yield case["case_id"], policy, f"run{index+1}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SCHEDULING)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if "," in args.gpu:
        raise ValueError("Use one GPU")
    lock_path = ROOT / "runs/tmp/h2_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another H2 runner is active")
    config, _, pilot, base, _, cases, _ = load_scheduling_inputs(args.config)
    directory = ROOT / config["results_dir"]
    journal_path = directory / "run.json"
    source_hashes = {name: digest(ROOT / name) for name in SCHEDULING_SOURCES}
    config_hash = digest(args.config)
    if journal_path.exists():
        old = json.loads(journal_path.read_text())
        if not args.resume or old["config_sha256"] != config_hash or old["source_sha256"] != source_hashes:
            raise SystemExit("Existing journal: inspect and resume unchanged sources, or use a fresh results_dir")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/triton"))
    python = str(ROOT / "runs/envs/nano-vllm/bin/python")
    logs = ROOT / "runs/logs/h_nanovllm" / directory.name
    logs.mkdir(parents=True, exist_ok=True)
    jobs = list(scheduling_jobs(config))
    if len(jobs) != 18:
        raise ValueError("Bounded pilot must have exactly 18 formal processes")
    report = dict(schema_version=1, stage="H2", scope="admission_policy", status="running", pid=os.getpid(),
        started_at_utc=now(), config_sha256=config_hash, source_sha256=source_hashes,
        planned_formal_processes=len(jobs), completed_formal_processes=0, completed_steps=[])

    def run(name, command):
        if digest(args.config) != config_hash or any(digest(ROOT / name) != checksum for name, checksum in source_hashes.items()):
            raise ValueError("Sources changed while running")
        report.update(current_step=name, updated_at_utc=now())
        write_result(report, journal_path)
        print(f"Scheduling [{report['completed_formal_processes']}/18]: {name}", flush=True)
        with (logs / f"{name.replace('/', '_')}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=900)
        report["completed_steps"].append(name)
        write_result(report, journal_path)

    try:
        gate_path = directory / "preflight.json"
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
        if not (args.resume and gate and gate["status"] == "pass"
                and gate["provenance"]["scheduling_config_sha256"] == config_hash
                and gate["provenance"]["scheduling_source_sha256"] == source_hashes):
            for policy in POLICIES:
                run(f"preflight/{policy}", [python, "-m", "serving.benchmark_h2_scheduling",
                    "--config", str(args.config), "--policy", policy, "--probe",
                    "--result", str(directory / "preflight" / f"{policy}.json")])
            run("preflight/compare", [python, "scripts/summarize_h2_scheduling.py", "--config", str(args.config), "--gate-only"])
            gate = json.loads(gate_path.read_text())
        if args.preflight_only:
            report.update(status="pass", current_step="preflight_complete")
        else:
            for case, policy, run_id in jobs:
                path = directory / "raw" / case / policy / f"{run_id}.json"
                if path.exists():
                    if not args.resume:
                        raise ValueError("Preserve existing raw evidence")
                    old = json.loads(path.read_text())
                    validate_run(old, config, pilot, base, cases, gate)
                    if old["qualification_gate_sha256"] != digest(gate_path):
                        raise ValueError("Existing raw result has a stale gate")
                else:
                    run(f"{case}/{policy}/{run_id}", [python, "-m", "serving.benchmark_h2_scheduling",
                        "--config", str(args.config), "--policy", policy, "--case-id", case,
                        "--run-id", run_id, "--result", str(path)])
                report["completed_formal_processes"] += 1
                write_result(report, journal_path)
            run("summarize_and_audit", [python, "scripts/summarize_h2_scheduling.py", "--config", str(args.config)])
            report.update(status="pass", current_step="complete")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"] = now()
        write_result(report, journal_path)
    print("Scheduling pilot:",report["status"],flush=True)


if __name__ == "__main__":
    main()

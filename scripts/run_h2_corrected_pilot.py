"""Run 36 isolated corrected-policy H2 processes, sequentially on one GPU."""

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
from serving.benchmark_h2_corrected import (
    DEFAULT_PERFORMANCE, PERFORMANCE_SOURCES, load_performance_inputs, validate_timed_run,
)
from serving.h2_runtime import now
from scripts.run_h2_nanovllm import pilot_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if "," in args.gpu:
        raise ValueError("Expose exactly one GPU")
    lock_path = ROOT / "runs/tmp/h2_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another H2 runner is active")
    inputs = load_performance_inputs(args.config)
    performance, _, _, config, _, cases, gate, gate_path = inputs
    gate_hash, config_hash = digest(gate_path), digest(args.config)
    source_hashes = {name: digest(ROOT / name) for name in PERFORMANCE_SOURCES}
    directory = ROOT / performance["results_dir"]
    journal_path = directory / "run.json"
    if journal_path.exists():
        old = json.loads(journal_path.read_text())
        if not args.resume:
            raise SystemExit("Journal exists; inspect it and use --resume")
        if (old["performance_config_sha256"] != config_hash
                or old["performance_source_sha256"] != source_hashes
                or old["qualification_gate_sha256"] != gate_hash):
            raise SystemExit("Existing journal has different sources or qualification; use a fresh results_dir")
    jobs = list(pilot_jobs({**config, "pilot_case_ids": performance["pilot_case_ids"],
        "independent_runs": performance["independent_runs"]}, performance["variants"]))
    if len(jobs) != 36:
        raise ValueError("Pilot must remain bounded to 36 processes")
    python = str(ROOT / "runs/envs/nano-vllm/bin/python")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/triton"))
    logs = ROOT / "runs/logs/h_nanovllm" / directory.name
    logs.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, stage="H2", status="running", pid=os.getpid(),
        started_at_utc=now(), qualification_gate_sha256=gate_hash,
        performance_config_sha256=config_hash, performance_source_sha256=source_hashes,
        planned_processes=len(jobs), completed_formal_processes=0, completed_steps=[],
        case_ids=performance["pilot_case_ids"], variants=performance["variants"],
        numerical_policy_id=gate["policy_id"], scope=performance["scope"])

    def run(name, command):
        if digest(args.config) != config_hash or any(digest(ROOT / p) != h for p, h in source_hashes.items()):
            raise RuntimeError("Performance sources changed during the pilot")
        report.update(current_step=name, updated_at_utc=now())
        write_result(report, journal_path)
        print(f"H2 corrected [{report['completed_formal_processes']}/36]: {name}", flush=True)
        with (logs / f"{name.replace('/', '_')}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=900, check=True)
        report["completed_steps"].append(name)

    try:
        for case_id, name, run_id in jobs:
            path = directory / "raw" / case_id / name / f"{run_id}.json"
            if path.exists():
                old = json.loads(path.read_text())
                if not args.resume or old.get("status") != "pass":
                    raise RuntimeError(f"Preserve existing raw evidence and inspect before retry: {path}")
                validate_timed_run(old, cases)
                if (old.get("qualification_gate_sha256") != gate_hash
                        or old["provenance"].get("performance_config_sha256") != config_hash
                        or old["provenance"].get("performance_source_sha256") != source_hashes
                        or any(old["provenance"].get(k) != v for k, v in gate["provenance"].items())):
                    raise ValueError("Existing raw evidence is stale; use a fresh results directory")
            else:
                run(f"{case_id}/{name}/{run_id}", [python, "-m", "serving.benchmark_h2_corrected",
                    "--config", str(args.config), "--case-id", case_id, "--variant", name,
                    "--run-id", run_id, "--result", str(path)])
            report["completed_formal_processes"] += 1
            write_result(report, journal_path)
        run("summarize", [python, "scripts/summarize_h2_nanovllm.py", "--raw-dir", str(directory / "raw"),
            "--output", str(directory / "summary.json")])
        run("audit", [python, "scripts/audit_h2_corrected_pilot.py", "--config", str(args.config)])
        report.update(status="pass", current_step="complete")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"] = now()
        write_result(report, journal_path)
    print(f"Corrected H2 pilot: pass -> {journal_path}", flush=True)


if __name__ == "__main__":
    main()

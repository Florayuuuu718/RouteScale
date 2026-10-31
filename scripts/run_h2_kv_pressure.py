"""Six numerical processes, exact gate, then 18 bounded serial GPU benchmarks."""

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
from serving.h2_kv_pressure import DEFAULT_CONFIG, POLICIES, SOURCES
from serving.h2_runtime import now


def formal_jobs(config):
    for index in range(1,4):
        policies = POLICIES if index % 2 else tuple(reversed(POLICIES))
        for case in config["cases"]:
            for policy in policies:
                yield case["case_id"], policy, f"run{index}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = ROOT / config["results_dir"]
    journal_path = directory / "run.json"
    source = {name:digest(ROOT/name) for name in SOURCES}
    if journal_path.exists():
        report = json.loads(journal_path.read_text())
        if not args.resume or report["config_sha256"] != digest(args.config) or report["source_sha256"] != source:
            raise SystemExit("Preserve existing attempt; --resume requires identical config and source")
        if report["status"] == "pass":
            raise SystemExit("Already complete; use the read-only summary audit instead")
    else:
        if any(directory.rglob("*.json")):
            raise SystemExit("Evidence exists without a journal; select a fresh results_dir")
        report = dict(schema_version=1, stage="H2", status="running", started_at_utc=now(),
            config_sha256=digest(args.config), source_sha256=source, completed_formal_processes=0,
            expected_formal_processes=18, completed_numerical_processes=0)
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/triton"))
    python = str(ROOT / "runs/envs/nano-vllm/bin/python")
    logs = ROOT / "runs/logs/h_nanovllm" / directory.name
    logs.mkdir(parents=True, exist_ok=True)
    report.update(pid=os.getpid(), status="running")

    def run(label, command):
        report.update(current_step=label, updated_at_utc=now())
        write_result(report, journal_path)
        print(f"H2 KV pressure: {label}", flush=True)
        with (logs/f"{label}.log").open("x") as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)

    try:
        gate_path = directory/"preflight.json"
        if not gate_path.exists():
            completed = 0
            for case in config["cases"]:
                for policy in POLICIES:
                    label = f"probe_{case['case_id']}_{policy}"
                    path = directory/"preflight"/f"{case['case_id']}_{policy}.json"
                    if path.exists():
                        if json.loads(path.read_text())["status"] != "pass":
                            raise ValueError("A failed probe cannot be overwritten by resume")
                    else:
                        run(label,[python,"-m","serving.benchmark_h2_kv_pressure","--config",str(args.config),
                            "--case-id",case["case_id"],"--policy",policy,"--probe","--result",str(path)])
                    completed += 1
                    report["completed_numerical_processes"] = completed
            run("qualify",[python,"scripts/summarize_h2_kv_pressure.py","--config",str(args.config),"--preflight"])
        gate = json.loads(gate_path.read_text())
        if gate["status"] != "pass" or not gate["gpu_performance_execution_allowed"]:
            raise ValueError("Failed numerical gate forbids formal timing")
        if (gate["contract"]["provenance"]["pressure_source_sha256"] != source
                or gate["contract"]["provenance"]["pressure_config_sha256"] != digest(args.config)):
            raise ValueError("Qualification source/config differs from this runner")
        for relative, checksum in gate["report_sha256"].items():
            if digest(directory/relative) != checksum:
                raise ValueError("Qualified numerical report changed")
        report["completed_numerical_processes"] = 6
        if args.preflight_only:
            report.update(status="preflight_pass", current_step="awaiting_formal_matrix")
        else:
            completed = 0
            for case,policy,run_id in formal_jobs(config):
                path = directory/"raw"/case/policy/f"{run_id}.json"
                if path.exists():
                    data = json.loads(path.read_text())
                    if (data["status"] != "pass" or data["qualification_gate_sha256"] != digest(gate_path)
                            or data["provenance"] != gate["contract"]["provenance"]):
                        raise ValueError("Failed/stale raw evidence must not be overwritten by resume")
                else:
                    run(f"{case}_{policy}_{run_id}",[python,"-m","serving.benchmark_h2_kv_pressure",
                        "--config",str(args.config),"--case-id",case,"--policy",policy,"--run-id",run_id,"--result",str(path)])
                completed += 1
                report["completed_formal_processes"] = completed
            run("audit",[python,"scripts/summarize_h2_kv_pressure.py","--config",str(args.config)])
            report.update(status="pass", current_step="complete")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"] = now()
        write_result(report,journal_path)


if __name__ == "__main__":
    main()

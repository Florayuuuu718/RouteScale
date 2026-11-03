"""Prepare F EP jobs or execute them on existing hardware; never rents GPUs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/f_ep_matrix.json"
SOURCES = ("expert_parallel.py", "moe.py", "model.py", "triton_grouped_gemm.py",
           "scripts/f_ep_worker.py", "scripts/audit_f_ep.py", "scripts/prepare_f_ep.py",
           "scripts/accept_f_ep_preparation.py", "config/f_ep_matrix.json",
           "docs/f_stage_protocol.md", "docs/f_ep_runbook.md")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jobs(local=True):
    if not local:
        return [dict(label=f"cuda{world}_{dtype}{'_gpt' if gpt else ''}", world=world, device="cuda", dtype=dtype,
                     formal=False, gpt=gpt, trace=False)
                for world in (1, 2) for dtype in ("float32", "bfloat16") for gpt in (False, True)]
    result = [dict(label=f"cpu{world}", world=world, device="cpu", dtype="float64") for world in (1, 2, 4)]
    result.append(dict(label="cpu2_gpt", world=2, device="cpu", dtype="float32", gpt=True))
    for dtype in ("float32", "bfloat16"):
        for mode in ("small", "formal", "gpt"):
            result.append(dict(label=f"cuda1_{dtype}_{mode}", world=1, device="cuda", dtype=dtype,
                               formal=mode == "formal", gpt=mode == "gpt"))
    result += [dict(label="cpu2_trace", world=2, device="cpu", dtype="float64", trace=True),
               dict(label="cuda1_trace", world=1, device="cuda", dtype="float32", trace=True)]
    return [{**dict(formal=False, gpt=False, trace=False), **job} for job in result]


def require_idle_gpus(world):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = [[int(v.strip()) for v in row.split(",")] for row in output.strip().splitlines()]
    if len(rows) < world:
        raise ValueError(f"requires {world} distinct GPUs, found {len(rows)}")
    if any(memory or utilization for _, memory, utilization in rows[:world]):
        raise ValueError("selected GPUs must be idle")
    return rows


def run_bounded(command, log, timeout):
    started = time.monotonic()
    timed_out = False
    with open(log, "x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            returncode = process.returncode
    return dict(returncode=returncode, timed_out=timed_out, wall_s=time.monotonic() - started)


def qualification_gate(directory):
    """A CPU or EP1 result cannot unlock the same-host multi-GPU timing scope."""
    directory = Path(directory)
    for job in jobs(local=False):
        path = directory / job["label"]
        manifest = json.loads((path / "job.json").read_text())
        audit = json.loads((path / "audit.json").read_text())
        if any(manifest.get(k) != job[k] for k in ("world", "device", "dtype", "formal", "gpt", "trace")):
            raise ValueError("wrong numerical qualification scope")
        if not audit.get("passed") or audit["job_sha256"] != sha(path / "job.json"):
            raise ValueError("failed or stale numerical qualification")
        if manifest["config_sha256"] != sha(CONFIG):
            raise ValueError("changed EP configuration")
        if set(manifest["source_sha256"]) != set(SOURCES):
            raise ValueError("incomplete EP source manifest")
        for name, digest in manifest["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("changed EP source")
        for name, digest in audit["raw_sha256"].items():
            if sha(name) != digest:
                raise ValueError("changed numerical tensors")
        for name, digest in audit["report_sha256"].items():
            if sha(path / name) != digest:
                raise ValueError("changed rank reports")
    return True


def run_job(job, output, python, config):
    output.mkdir(parents=True, exist_ok=False)
    lock = None
    gpu_before = None
    try:
        if job["device"] == "cuda":
            lock_path = ROOT / "runs/tmp/h2_runner.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = lock_path.open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu_before = require_idle_gpus(job["world"])
        command = [str(Path(python).absolute()), "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={job['world']}", "--max-restarts=0",
                   "scripts/f_ep_worker.py", "--device", job["device"], "--dtype", job["dtype"], "--output", str(output.resolve())]
        for flag in ("formal", "gpt", "trace"):
            if job[flag]:
                command.append("--" + flag)
        source = {name: sha(ROOT / name) for name in SOURCES}
        result = run_bounded(command, output / "worker.log", config["formal_job_timeout_s"] if job["formal"] else config["job_timeout_s"])
        manifest = {**job, **result, "command": command, "source_sha256": source,
                    "config_sha256": sha(CONFIG), "gpu_before": gpu_before,
                    "host": os.uname().nodename, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        if job["device"] == "cuda":
            # Process termination precedes sampling. No zero values are fabricated.
            manifest["gpu_after"] = require_idle_gpus(job["world"])
            manifest["topology"] = subprocess.run(["nvidia-smi", "topo", "-m"], text=True, capture_output=True, timeout=15).stdout
        (output / "job.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if result["returncode"] != 0:
            raise RuntimeError(f"worker failed; retained {output / 'worker.log'}")
        audit_command = [str(Path(python).absolute()), "scripts/audit_f_ep.py", str(output)]
        audited = run_bounded(audit_command, output / "audit.log", config["formal_job_timeout_s"])
        if audited["returncode"] != 0:
            raise RuntimeError(f"independent audit failed; retained {output / 'audit.log'}")
        return json.loads((output / "audit.json").read_text())
    finally:
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host-qualification", action="store_true")
    parser.add_argument("--labels", nargs="*")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    selected = jobs(local=not args.host_qualification)
    if args.labels:
        unknown = set(args.labels) - {job["label"] for job in selected}
        if unknown:
            raise ValueError(f"unknown jobs: {unknown}")
        selected = [job for job in selected if job["label"] in args.labels]
    if not args.execute:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"scope": config["scope"], "paid_resources_authorized": False,
                       "jobs": selected, "pending_host_qualifications": jobs(local=False),
                       "performance_execution_allowed": False}, stream, indent=2)
        print(str(args.output))
        return
    for job in selected:
        print(f"start {job['label']}", flush=True)
        audit = run_job(job, args.output / job["label"], args.python, config)
        print(json.dumps({"job": job["label"], "passed": audit["passed"], "checks": audit["checks"]}), flush=True)


if __name__ == "__main__":
    main()

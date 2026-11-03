"""Prepare/run F complete training qualification and gated same-host timing."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_f_ep import require_idle_gpus, run_bounded, sha, qualification_gate, jobs as module_jobs

SOURCES = ("ep_training.py", "expert_parallel.py", "moe.py", "model.py", "triton_grouped_gemm.py",
           "scripts/f_ep_training.py", "scripts/audit_f_ep_training.py", "scripts/prepare_f_training.py",
           "scripts/f_ep_worker.py", "scripts/prepare_f_ep.py", "config/f_ep_rental_requirements.json",
           "docs/f_training_preparation.md")


def jobs(host=False, benchmark=False):
    if benchmark:
        return [dict(label=f"{mode}_run{run}", mode=mode, world=1 if mode == "reference" else 2,
                     device="cuda", dtype="bfloat16", tiny=False)
                for run in (1, 2, 3) for mode in ("reference", "replicated", "ep")]
    result = [dict(label="cpu2_tiny", world=2, device="cpu", dtype="float32", tiny=True),
              dict(label="cuda1_tiny_fp32", world=1, device="cuda", dtype="float32", tiny=True),
              dict(label="cuda1_tiny_bf16", world=1, device="cuda", dtype="bfloat16", tiny=True),
              dict(label="cuda1_full", world=1, device="cuda", dtype="bfloat16", tiny=False)]
    if host:
        result.append(dict(label="cuda2_full", world=2, device="cuda", dtype="bfloat16", tiny=False))
    return result


def performance_gate(host_root):
    host_root = Path(host_root)
    qualification_gate(host_root / "module")
    for module_job in module_jobs(False):
        manifest = json.loads((host_root / "module" / module_job["label"] / "job.json").read_text())
        if manifest["host"] != os.uname().nodename:
            raise ValueError("module qualification must be on this same host")
    for label, world in (("cuda1_full", 1), ("cuda2_full", 2)):
        path = host_root / "training" / label
        job = json.loads((path / "job.json").read_text())
        result = json.loads((path / "audit.json").read_text())
        if not result["passed"] or job["world"] != world or job["device"] != "cuda" or job["tiny"] or job["dtype"] != "bfloat16":
            raise ValueError("complete same-host CUDA1/2 qualification required")
        if job["host"] != os.uname().nodename or result["job_sha256"] != sha(path / "job.json"):
            raise ValueError("different host or changed qualification")
        if set(job["source_sha256"]) != set(SOURCES):
            raise ValueError("incomplete training source manifest")
        for name, digest in job["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("changed training source")
        for name, digest in result["raw_sha256"].items():
            if sha(name) != digest:
                raise ValueError("changed training tensors")
        for name, digest in result["report_sha256"].items():
            if sha(path / name) != digest:
                raise ValueError("changed training rank reports")
    return True


def run(job, output, python, host_root):
    output.mkdir(parents=True, exist_ok=False)
    lock = None
    try:
        if job["device"] == "cuda":
            lock = (ROOT / "runs/tmp/h2_runner.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require_idle_gpus(job["world"])
        if "mode" in job:
            performance_gate(host_root)
        command = [str(Path(python).absolute()), "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={job['world']}", "--max-restarts=0", "scripts/f_ep_training.py",
                   "--device", job["device"], "--dtype", job["dtype"], "--output", str(output.resolve())]
        if job["tiny"]:
            command.append("--tiny")
        if "mode" in job:
            command += ["--benchmark", job["mode"]]
        source = {name: sha(ROOT / name) for name in SOURCES}
        outcome = run_bounded(command, output / "worker.log", 900 if "mode" in job else 600)
        manifest = {**job, **outcome, "host": os.uname().nodename, "command": command,
                    "source_sha256": source, "gpu_after": require_idle_gpus(job["world"]) if job["device"] == "cuda" else None}
        (output / "job.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if outcome["returncode"]:
            raise RuntimeError(f"training worker failed: {output}")
        if "mode" not in job:
            outcome = run_bounded([str(Path(python).absolute()), "scripts/audit_f_ep_training.py", str(output)], output / "audit.log", 600)
            if outcome["returncode"]:
                raise RuntimeError(f"training tensor audit failed: {output}")
            result = json.loads((output / "audit.json").read_text())
            print(json.dumps({"label": job["label"], "passed": result["passed"], "checks": result["checks"]}), flush=True)
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host-qualification", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--host-root", type=Path)
    parser.add_argument("--labels", nargs="*")
    args = parser.parse_args()
    selected = jobs(args.host_qualification, args.benchmark)
    if args.labels:
        if set(args.labels) - {job["label"] for job in selected}:
            raise ValueError("unknown training jobs")
        selected = [job for job in selected if job["label"] in args.labels]
    if not args.execute:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"jobs": selected, "paid_resources_authorized": False, "performance_execution_allowed": False}, stream, indent=2)
    else:
        if args.benchmark and not args.host_root:
            raise ValueError("benchmark requires --host-root and complete same-host gates")
        for job in selected:
            print("start " + job["label"], flush=True)
            run(job, args.output / job["label"], args.python, args.host_root)

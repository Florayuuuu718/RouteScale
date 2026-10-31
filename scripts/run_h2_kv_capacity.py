"""Bounded serial capacity validation; preserves all existing experiment outputs."""

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
from serving.verify_h2_kv_capacity import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    directory = ROOT / cfg["results_dir"]
    if any((directory / name).exists() for name in ("run.json", "static_window.json", "paged.json", "audit.json")):
        raise SystemExit("Existing evidence is immutable; select a fresh results_dir")
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/triton"))
    python = str(ROOT / "runs/envs/nano-vllm/bin/python")
    logs = ROOT / "runs/logs/h_nanovllm" / directory.name
    logs.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, stage="H2", status="running", pid=os.getpid(), started_at_utc=now(),
        config_sha256=digest(args.config), completed_steps=[])
    try:
        for name in ("static_window", "paged", "audit"):
            report.update(current_step=name, updated_at_utc=now())
            write_result(report, directory / "run.json")
            command = [python, "-m", "serving.verify_h2_kv_capacity", "--config", str(args.config)]
            if name != "audit":
                command += ["--policy", name]
            print(f"H2 KV capacity: {name}", flush=True)
            with (logs / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=900)
            report["completed_steps"].append(name)
        report.update(status="pass", current_step="complete")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"] = now()
        write_result(report, directory / "run.json")


if __name__ == "__main__":
    main()

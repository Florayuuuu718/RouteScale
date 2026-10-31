"""Serial I2 subprocess controller; numerical and service gates precede timing."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.i2_workloads import CONFIG, digest
from scripts.prepare_i2_minisglang import prepare_i2
from scripts.run_i1_repaired import living_children


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def environment(config):
    toolkit = ROOT / "runs/toolchains/cuda-13.0-i1"
    return dict(os.environ, PATH=str(Path(sys.executable).parent) + ":" + str(toolkit / "bin") + ":" + os.environ["PATH"],
        CUDA_HOME=str(toolkit), CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", MAX_JOBS="4",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
        MINISGL_DISABLE_OVERLAP_SCHEDULING="1", ROUTESCALE_I2_PARENT_LOCKED="1",
        TVM_FFI_CACHE_DIR=str(ROOT / "runs/cache/i_minisglang/tvm"),
        FLASHINFER_WORKSPACE_BASE=str(ROOT / "runs/cache/i_minisglang/flashinfer"))


def launch(command, label, config, attempt, output, journal):
    import psutil
    raw = ROOT / config["runtime_dir"] / attempt
    raw.mkdir(parents=True, exist_ok=True)
    record = dict(label=label, command=command, status="running", wall_started_ns=time.time_ns())
    journal.append(record)
    write(output / "processes.json", journal)
    logpath = raw / f"{label}.log"
    started = time.monotonic()
    with logpath.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment(config), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = process.pid
        try:
            result = process.wait(timeout=config["subprocess_timeout_s"])
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            record["status"] = "forced_cleanup"
            record["returncode"] = process.returncode
            record["seconds"] = time.monotonic() - started
            write(output / "processes.json", journal)
            raise
    record.update(status="pass" if result == 0 else "fail", returncode=result,
        seconds=time.monotonic() - started, log=str(logpath.relative_to(ROOT)), log_sha256=digest(logpath),
        gpu_after=subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], text=True).strip())
    write(output / "processes.json", journal)
    print(json.dumps(record), flush=True)
    if result != 0:
        raise RuntimeError(f"{label} failed; preserve {logpath}; choose a new attempt after repairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--phase", choices=["reference", "numerics", "service", "pilot", "benchmark", "all"], required=True)
    parser.add_argument("--variant")
    args = parser.parse_args()
    if not args.attempt.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Simple attempt name required")
    config = json.loads(CONFIG.read_text())
    if Path(sys.prefix).resolve() != (ROOT / config["environment"]).resolve():
        raise ValueError("Use the isolated Mini-SGLang environment")
    series = prepare_i2()
    output = ROOT / config["results_dir"] / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [str(p.relative_to(ROOT)) for p in [*sorted((ROOT / "serving").glob("i2_*.py")),
        *sorted((ROOT / "scripts").glob("*i2*.py")), *sorted((ROOT / "tests").glob("test_i2*.py")),
        CONFIG, *sorted((ROOT / "patches/minisgl-i2").glob("*"))] if p.is_file()]
    source_hashes = {name: digest(ROOT / name) for name in sources}
    meta_path = output / "provenance.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["source_sha256"] != source_hashes or old["series"] != series:
            raise ValueError("Preserve frozen attempt sources; choose a new attempt")
    else:
        for name in sources:
            dest = output / "source_snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        write(output / "workload.json", json.loads((ROOT / config["workload_path"]).read_text()))
        write(meta_path, dict(config=config, source_sha256=source_hashes, series=series,
            model_sha256={p.name:digest(p) for p in (ROOT / config["model_path"]).iterdir() if p.is_file()},
            workload_sha256=digest(ROOT / config["workload_path"]),
            upstream_commit="9a91cfafe754aa85daee49998176275667eb58f2", paid_resources_used=False))
    journal_path = output / "processes.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else []
    if any(row["status"] != "pass" for row in journal):
        raise ValueError("Attempt contains an interrupted or failed process; use a fresh attempt")

    def worker(mode, variant=None, run=0):
        label = "reference" if mode == "reference" else f"{mode}_{variant}_run{run}"
        if (output / f"{label}.json").exists():
            row = json.loads((output / f"{label}.json").read_text())
            if row["status"] != "pass":
                raise ValueError("Do not reuse failed evidence")
            return
        command = [sys.executable, "-m", "serving.i2_worker", "--attempt", args.attempt, "--mode", mode, "--run", str(run)]
        if variant:
            command += ["--variant", variant]
        launch(command, label, config, args.attempt, output, journal)

    if args.phase in {"reference", "numerics", "all"}:
        worker("reference")
    if args.phase in {"numerics", "all"}:
        for variant in ([args.variant] if args.variant else config["variants"]):
            worker("numeric", variant)
        if not args.variant:
            from scripts.audit_i2_numerics import audit_numerics
            gate = audit_numerics(output)
            write(output / "numerical_gate.json", gate)
            if not gate["passed"]:
                raise ValueError("Numerical gate failed; timing remains locked")
    if args.phase in {"service", "all"}:
        from scripts.run_i2_service_gate import run_service_gate
        run_service_gate(config, args.attempt, output, journal, launch)
    if args.phase in {"pilot", "benchmark", "all"}:
        for gate in ("numerical_gate.json", "service_gate.json"):
            if not json.loads((output / gate).read_text())["passed"]:
                raise ValueError("Numerical and service gates must pass before timing")
        names = list(config["variants"])
        runs = [0] if args.phase == "pilot" else range(1, config["independent_runs"] + 1)
        for run in runs:
            order = names if run == 1 else list(reversed(names)) if run == 2 else names[3:] + names[:3]
            for name in ([args.variant] if args.variant else order):
                worker("benchmark", name, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

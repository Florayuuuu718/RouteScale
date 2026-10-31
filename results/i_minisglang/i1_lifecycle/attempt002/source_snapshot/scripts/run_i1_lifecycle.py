"""Reproduce the observed, unmodified Mini-SGLang I1 baseline in a fresh attempt."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.online_client import OnlineClient, OnlineRequest, canonical_hash


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def events(directory):
    result = []
    for path in directory.glob("events.*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:  # a concurrent final append may be incomplete
                continue
    return sorted(result, key=lambda row: row["monotonic_ns"])


def memory():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                                    "--format=csv,noheader"], text=True).strip()


def server_command(config):
    args = [sys.executable, str(ROOT / "scripts/launch_i1_minisglang.py"), "--model",
            str(ROOT / config["model"]["path"])]
    for key, value in config["server"].items():
        if key == "overlap_scheduling":
            continue
        args += ["--" + key.replace("_", "-"), str(value)]
    return args


async def exercise(config, event_dir, output):
    import httpx
    from transformers import AutoTokenizer

    base = f"http://{config['server']['host']}:{config['server']['port']}"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model"]["path"], local_files_only=True)
    ledger, workload = [], []
    def sink(row):
        with (event_dir / "client.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    client = OnlineClient(base, concurrency=config["client"]["concurrency"], sink=sink)

    def request(rid, message="Reply with the single word OK.", max_tokens=128, **options):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": message}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt)
        payload = dict(model=str(ROOT / config["model"]["path"]), prompt=prompt,
                       temperature=0, top_p=1, top_k=1, max_tokens=max_tokens,
                       ignore_eos=options.pop("ignore_eos", False))
        workload.append(dict(request_id=rid, message=message, prompt=prompt, input_ids=ids,
            input_ids_sha256=canonical_hash(ids), payload=payload, payload_sha256=canonical_hash(payload),
            options=dict(options)))
        return OnlineRequest(rid, payload, timeout_s=options.pop("timeout_s", 30), **options)

    async def run(reqs):
        records = await client.run(reqs)
        ledger.extend(records)
        write(output / "requests.json", ledger)
        write(output / "workload.json", dict(chat_template_sha256=canonical_hash(tokenizer.chat_template),
              enable_thinking=False, eos_token_id=tokenizer.eos_token_id, requests=workload))
        return records

    snapshots = []
    async with httpx.AsyncClient(base_url=base, timeout=2, trust_env=False) as http:
        async def snapshot(label):
            response = await http.get("/debug/i1/frontend")
            response.raise_for_status()
            snapshots.append(dict(label=label, monotonic_ns=time.monotonic_ns(), **response.json()))
            write(output / "frontend_snapshots.json", snapshots)

        async def settle(label):
            # Frozen upper bound, not a performance warmup. Evidence remains raw events.
            await asyncio.sleep(config["budgets"]["cleanup_s"])
            await snapshot(label)

        await snapshot("initial")
        first, = await run([request("eos_short", timeout_s=config["budgets"].get("first_request_s", 120))])
        if first["state"] != "finished":
            raise RuntimeError("First real-kernel request did not finish; stop before further cases")
        await settle("after_eos")
        await run([request("max_tokens_32", message="Count the positive integers in order.",
                           max_tokens=32, ignore_eos=True)])
        await settle("after_length")
        await run([request(f"concurrent_{i}", message=f"Compute {i}+2. Reply briefly.", max_tokens=48)
                   for i in range(3)])
        await settle("after_concurrent")

        # Same run origin; second arrives while the first holds the only running slot.
        await run([request("queue_blocker", message="List successive integers, separated by commas.",
                           max_tokens=384, ignore_eos=True),
                   request("queued_cancel", max_tokens=64, arrival_s=.1, cancel_after_s=.3)])
        await settle("after_queued_cancel")
        await run([request("decode_disconnect", message="Write a long numbered list of integers.",
                           max_tokens=384, ignore_eos=True, disconnect_after_chunks=4)])
        await settle("after_disconnect")
        await run([request("decode_deadline", message="Write a long numbered list of integers.",
                           max_tokens=384, ignore_eos=True, timeout_s=.3)])
        await settle("after_deadline")
        # A further normal request makes retained detokenizer UIDs observable.
        await run([request("post_cancel_probe", max_tokens=24)])
        await settle("after_post_cancel_probe")

        bad = request("invalid_type")
        workload[-1]["payload"]["max_tokens"] = "invalid"
        workload[-1]["payload_sha256"] = canonical_hash(workload[-1]["payload"])
        await run([OnlineRequest(bad.request_id, {**bad.payload, "max_tokens": "invalid"})])
        # Deliberately exceeds the configured window. Deadline is finite even on silent drop.
        await run([request("context_overflow", message="word " * 1100, max_tokens=16, timeout_s=1)])
        await settle("after_validation")
        # Repeat the external ID in separate submissions: observe upstream behavior.
        await run([request("repeated_header_1", max_tokens=8, external_id="repeated-external-id")])
        await run([request("repeated_header_2", max_tokens=8, external_id="repeated-external-id")])
        await settle("after_repeated_external_id")

        # Dedicated worker failure, last so earlier evidence is not contaminated.
        rows = events(event_dir)
        schedulers = {row["pid"] for row in rows if row["event"] == "scheduler_token"}
        if len(schedulers) != 1:
            raise ValueError(f"Expected one GPU scheduler; observed {schedulers}")
        worker = schedulers.pop()
        fault_at = time.monotonic_ns()
        os.kill(worker, signal.SIGTERM)
        await run([request("worker_failure", max_tokens=8,
                           timeout_s=config["budgets"]["worker_failure_s"])])
        health = await http.get("/v1")
        write(output / "worker_failure.json", dict(injected_signal="SIGTERM", worker_pid=worker,
            monotonic_ns=fault_at, frontend_health_status=health.status_code,
            frontend_health_body=health.json(), automatic_recovery_tested=False))
        await snapshot("after_worker_failure")
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/serving/i1_minisglang.json")
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if not args.attempt.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Simple attempt name required")
    config = json.loads(args.config.read_text())
    output = ROOT / config["results_dir"] / args.attempt
    if output.exists():
        raise ValueError("Preserve existing attempt; choose a new name")
    vendor = ROOT / config["upstream"]["path"]
    actual = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if actual != config["upstream"]["commit"] or subprocess.check_output(
            ["git", "-C", str(vendor), "status", "--porcelain"], text=True).strip():
        raise ValueError("Pinned, clean upstream checkout required")
    expected_python = ROOT / config["environment"]["path"] / "bin/python"
    if Path(sys.prefix).resolve() != expected_python.parent.parent.resolve():
        raise ValueError(f"Use isolated environment: {expected_python}")
    (ROOT / "runs/tmp").mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    event_dir = ROOT / config["runtime_dir"] / args.attempt
    event_dir.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True)
    files = ["serving/online_client.py", "serving/minisgl_observer.py", "scripts/run_i1_lifecycle.py",
             "scripts/launch_i1_minisglang.py", "tests/test_online_client.py", "docs/i1_lifecycle_protocol.md",
             "config/serving/i1_minisglang.json", "config/serving/i1_requirements.txt"]
    source_hashes = {}
    for name in files:
        source = ROOT / name
        if not source.is_file():
            continue
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[name] = sha(target)
    write(output / "source_snapshot/manifest.json", source_hashes)
    model_hashes = {p.name: sha(p) for p in (ROOT / config["model"]["path"]).iterdir() if p.is_file()}
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    provenance = dict(config=config, config_sha256=sha(args.config), upstream_commit=actual,
        model_files_sha256=model_hashes, python=sys.version, packages=packages,
        server_command=server_command(config), gpu_before=memory(), source_sha256=source_hashes)
    write(output / "provenance.json", provenance)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + ":/usr/local/cuda-13.0/bin:" + os.environ["PATH"],
        CUDA_HOME="/usr/local/cuda-13.0", CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1", MAX_JOBS="4",
        MINISGL_DISABLE_OVERLAP_SCHEDULING="1", ROUTESCALE_I1_EVENTS=str(event_dir),
        TVM_FFI_CACHE_DIR=str(ROOT / "runs/cache/i_minisglang/tvm"),
        FLASHINFER_WORKSPACE_BASE=str(ROOT / "runs/cache/i_minisglang/flashinfer"))
    record = dict(stage="I1", attempt=args.attempt, status="fail", evidence_class="functional_not_performance",
                  runtime_dir=str(event_dir.relative_to(ROOT)), start_wall_time_ns=time.time_ns())
    process = None
    try:
        import httpx
        started = time.monotonic()
        log = (event_dir / "server.log").open("w")
        process = subprocess.Popen(server_command(config), cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record["server_pid"] = process.pid
        base = f"http://{config['server']['host']}:{config['server']['port']}"
        with httpx.Client(timeout=1, trust_env=False) as http:
            while time.monotonic() - started < config["budgets"]["startup_s"]:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode} during startup")
                try:
                    if (http.get(base + "/v1/models").status_code == 200 and
                        http.get(base + "/debug/i1/frontend").json()["pid"] == process.pid):
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("Startup budget exceeded; see preserved server.log")
        record["startup_seconds"] = time.monotonic() - started
        record["gpu_serving"] = memory()
        records = asyncio.run(exercise(config, event_dir, output))
        record["client_terminal_counts"] = dict(Counter(r["state"] for r in records))
        record["status"] = "baseline_completed_pending_lifecycle_audit"
    except (Exception, KeyboardInterrupt) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        (event_dir / "runner_error.log").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        if process is not None:
            import psutil
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []
            tracked = [process.pid] + [p.pid for p in descendants]
            cleanup_start = time.monotonic()
            forced = False
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=config["budgets"]["shutdown_s"])
            except subprocess.TimeoutExpired:
                forced = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            alive = [p.pid for p in descendants if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            if alive:
                forced = True
                for pid in alive:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                psutil.wait_procs(descendants, timeout=3)
            record["shutdown"] = dict(seconds=time.monotonic() - cleanup_start,
                forced_cleanup=forced, returncode=process.returncode, tracked_pids=tracked,
                surviving_pids=[p.pid for p in descendants if p.is_running() and p.status() != psutil.STATUS_ZOMBIE])
            log.close()
        record["gpu_after"] = memory()
        rows = events(event_dir)
        write(output / "server_events.json", rows)
        record["raw_log_sha256"] = {p.name: sha(p) for p in event_dir.glob("*") if p.is_file()}
        write(output / "run.json", record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    return 0 if record["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())

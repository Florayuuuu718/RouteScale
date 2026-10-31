"""Run one isolated, real single-GPU lifecycle scenario on the replayed sources."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import fcntl
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
from scripts.prepare_i1_lifecycle import prepare, git
from scripts.run_i1_lifecycle import events, memory, sha, write
from serving.online_client import OnlineClient, OnlineRequest, canonical_hash


def living_children(children):
    import psutil
    alive = []
    for child in children:
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                alive.append(child)
        except psutil.NoSuchProcess:
            pass  # normal exit can race with the two liveness reads
    return alive


def command(config):
    result = [sys.executable, str(ROOT / "scripts/launch_i1_repaired.py"), "--model",
              str(ROOT / config["model"]["path"])]
    for key, value in config["server"].items():
        if key != "overlap_scheduling":
            result.extend(["--" + key.replace("_", "-"), str(value)])
    return result


async def exercise(config, event_dir, output, scenario, process):
    import httpx
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model"]["path"], local_files_only=True)
    base = f"http://{config['server']['host']}:{config['server']['port']}"
    ledger, workload, snapshots, settlements = [], [], [], []

    def sink(row):
        with (event_dir / "client.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    client = OnlineClient(base, config["client"]["concurrency"], sink=sink)

    def request(rid, message="Reply with the single word OK.", max_tokens=64, raw_prompt=None, **options):
        prompt = raw_prompt if raw_prompt is not None else tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt)
        payload = dict(model=str(ROOT / config["model"]["path"]), prompt=prompt,
            temperature=0, top_p=1, top_k=1, max_tokens=max_tokens,
            ignore_eos=options.pop("ignore_eos", False))
        workload.append(dict(request_id=rid, message=message, prompt=prompt, input_ids=ids,
            input_ids_sha256=canonical_hash(ids), payload=payload, payload_sha256=canonical_hash(payload),
            options=dict(options)))
        return OnlineRequest(rid, payload, timeout_s=options.pop("timeout_s", 30), **options)

    def save():
        write(output / "requests.json", ledger)
        write(output / "workload.json", dict(enable_thinking=False, eos_token_id=tokenizer.eos_token_id,
            chat_template_sha256=canonical_hash(tokenizer.chat_template), requests=workload))

    async def run(reqs):
        save()  # preserve all offered requests even if exercise is interrupted
        records = await client.run(reqs)
        ledger.extend(records)
        save()
        return records

    async with httpx.AsyncClient(base_url=base, timeout=2, trust_env=False) as http:
        async def snapshot(label):
            response = await http.get("/debug/i1/frontend")
            response.raise_for_status()
            row = dict(label=label, monotonic_ns=time.monotonic_ns(), **response.json())
            snapshots.append(row)
            write(output / "frontend_snapshots.json", snapshots)
            return row

        async def settle(label):
            started = time.monotonic_ns()
            deadline = time.monotonic() + config["budgets"]["cleanup_s"]
            while True:
                snap = await snapshot(label)
                rows = events(event_dir)
                idle = next((r for r in reversed(rows) if r["event"] == "scheduler_idle"), None)
                detok = next((r for r in reversed(rows) if r["event"] in {"detokenizer_state", "detokenizer_end"}), None)
                clear = all(not snap[k] for k in ("ack_uids", "event_uids", "deadline_uids",
                                                  "admitted_uids", "aborting_uids"))
                if clear and idle and not idle["pending_uids"] and not idle["running_uids"] and detok and not detok["active_uids"]:
                    settlements.append(dict(label=label, started_ns=started,
                        ended_ns=time.monotonic_ns(), snapshot_ns=snap["monotonic_ns"],
                        idle_ns=idle["monotonic_ns"], detokenizer_ns=detok["monotonic_ns"]))
                    write(output / "settlements.json", settlements)
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Resources did not settle after {label}")
                await asyncio.sleep(0.05)

        async def wait_running(rid, min_tokens=4):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                rows = events(event_dir)
                assigned = [r["uid"] for r in rows if r["event"] == "frontend_new_user" and r["external_id"] == rid]
                if assigned and sum(r["event"] == "scheduler_token" and r.get("uid") == assigned[0] for r in rows) >= min_tokens:
                    return assigned[0]
                await asyncio.sleep(0.02)
            raise TimeoutError(f"No real output observed for {rid}")

        async def wait_pending(rid):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                rows = events(event_dir)
                assigned = [r["uid"] for r in rows if r["event"] == "frontend_new_user" and r["external_id"] == rid]
                if assigned and any(r["event"] == "scheduler_state" and assigned[0] in r["pending_uids"] and
                                    r["running_uids"] for r in rows):
                    return
                await asyncio.sleep(0.02)
            raise TimeoutError(f"No pending backend request observed for {rid}")

        await snapshot("initial")
        first, = await run([request("eos_short", timeout_s=config["budgets"]["first_request_s"])])
        if first["state"] != "finished":
            raise RuntimeError("First real-kernel request failed; preserve this attempt")
        await settle("after_eos")

        if scenario != "healthy":
            requests = [request("fault_inflight", message="List successive integers separated by commas.",
                max_tokens=768, ignore_eos=True, timeout_s=15),
                request("fault_queued", max_tokens=32, timeout_s=15)]
            # Concurrent HTTP submissions may reach the tokenizer in either order.
            # Establish actual running/pending states before injecting the fault.
            task = asyncio.create_task(run([requests[0]]))
            await wait_running("fault_inflight")
            queued_task = asyncio.create_task(run([requests[1]]))
            await wait_pending("fault_queued")
            rows = events(event_dir)
            if scenario == "shutdown_active":
                target, sig = process.pid, signal.SIGTERM
            else:
                kind = "engine_ready" if scenario == "scheduler_fault" else "tokenized"
                targets = {r["pid"] for r in rows if r["event"] == kind}
                if len(targets) != 1:
                    raise ValueError(f"Ambiguous dedicated {kind} process: {targets}")
                target, sig = targets.pop(), signal.SIGTERM
            fault = dict(scenario=scenario, target_pid=target, signal=sig.name,
                         monotonic_ns=time.monotonic_ns(), automatic_recovery_tested=False)
            write(output / "scenario.json", fault)
            os.kill(target, sig)
            await asyncio.gather(task, queued_task)
            if scenario != "shutdown_active":
                await run([request("post_fault_admission", timeout_s=3)])
                health = await http.get("/ready")
                fault.update(health_status=health.status_code, health_body=health.json(),
                             health_monotonic_ns=time.monotonic_ns())
                deadline = time.monotonic() + config["budgets"]["cleanup_s"]
                while True:
                    snap = await snapshot("after_fault")
                    if all(not snap[k] for k in ("ack_uids", "event_uids", "deadline_uids", "admitted_uids", "aborting_uids", "backend_done_uids")):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Fault cleanup retained frontend admission/reply state")
                    await asyncio.sleep(0.05)
                fault["frontend_cleanup_ns"] = snap["monotonic_ns"]
                write(output / "scenario.json", fault)
            return ledger

        await run([request("max_tokens_32", message="Count the positive integers in order.", max_tokens=32, ignore_eos=True)])
        await settle("after_length")
        await run([request(f"concurrent_{i}", message=f"Compute {i}+2. Reply briefly.", max_tokens=48) for i in range(3)])
        await settle("after_concurrent")
        await run([request("queue_blocker", message="List successive integers separated by commas.", max_tokens=384, ignore_eos=True),
                   request("queued_cancel", arrival_s=0.1, cancel_after_s=0.25)])
        await settle("after_queued_cancel")
        await run([request("decode_disconnect", message="Write a long list of integers.", max_tokens=384,
                           ignore_eos=True, disconnect_after_chunks=4)])
        await settle("after_disconnect")
        await run([request("decode_deadline", message="Write a long list of integers.", max_tokens=384,
                           ignore_eos=True, timeout_s=0.3)])
        await settle("after_deadline")
        await run([request("nonstream_cancel", message="Write a long list of integers.", max_tokens=384,
                           ignore_eos=True, cancel_after_s=0.3, stream=False)])
        await settle("after_nonstream_cancel")
        await run([request("nonstream_eos", stream=False)])
        await run([request("nonstream_length", max_tokens=8, ignore_eos=True, stream=False)])
        await settle("after_nonstream")
        await run([request("post_cancel_probe", max_tokens=24)])
        bad = request("invalid_type")
        bad.payload["max_tokens"] = "invalid"
        workload[-1]["payload_sha256"] = canonical_hash(bad.payload)
        await run([bad])
        await run([request("zero_budget", max_tokens=0)])
        await run([request("context_overflow", message="word " * 1100, max_tokens=16)])
        context = config["server"]["max_seq_len_override"]
        for name, length in (("context_equal", context), ("context_last_slot", context - 1)):
            prompt = tokenizer.decode(tokenizer.encode("word " * (context + 8))[:length])
            if len(tokenizer.encode(prompt)) != length:
                raise ValueError("Exact boundary prompt does not round-trip")
            await run([request(name, raw_prompt=prompt, max_tokens=8, ignore_eos=True)])
        await settle("after_validation")
        await run([request("repeated_header_1", max_tokens=8, external_id="repeated-external-id")])
        await run([request("repeated_header_2", max_tokens=8, external_id="repeated-external-id")])
        await settle("after_repeated_header")
        # All arrivals exceed admission=4 while the only GPU slot is occupied.
        await run([request(f"overload_{i}", message="Count integers in order.", max_tokens=64,
                           ignore_eos=True) for i in range(12)])
        await settle("after_overload")
        await run([request("post_overload_probe", max_tokens=8)])
        await settle("final_idle")
        write(output / "scenario.json", dict(scenario=scenario, automatic_recovery_tested=False))
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/serving/i1_repaired.json")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--scenario", choices=["healthy", "scheduler_fault", "tokenizer_fault", "shutdown_active"], required=True)
    args = parser.parse_args()
    if not args.attempt.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Simple fresh attempt name required")
    config = json.loads(args.config.read_text())
    checkout = ROOT / config["repairs"]["checkout"]
    series = prepare(checkout)
    baseline = ROOT / config["upstream"]["path"]
    if git(baseline, "rev-parse", "HEAD") != series["base_commit"] or git(baseline, "status", "--porcelain"):
        raise ValueError("Pinned upstream must remain clean")
    expected = (ROOT / config["environment"]["path"]).resolve()
    if Path(sys.prefix).resolve() != expected:
        raise ValueError(f"Use {expected}/bin/python")
    output = ROOT / config["results_dir"] / args.attempt
    runtime = ROOT / config["runtime_dir"] / args.attempt
    if output.exists() or runtime.exists():
        raise ValueError("Do not overwrite an attempt or its logs")
    (ROOT / "runs/tmp").mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True)
    runtime.mkdir(parents=True)
    sources = ["scripts/run_i1_repaired.py", "scripts/audit_i1_repaired.py", "scripts/launch_i1_repaired.py",
        "scripts/prepare_i1_lifecycle.py", "scripts/run_i1_lifecycle.py", "serving/online_client.py",
        "serving/minisgl_observer.py", "tests/test_i1_lifecycle_repairs.py", "tests/test_online_client.py",
        str(args.config.relative_to(ROOT)), "config/serving/i1_requirements.txt"]
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "patches/minisgl-i1").iterdir()) if p.is_file()]
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    manifest = {name: sha(output / "source_snapshot" / name) for name in sources}
    write(output / "source_snapshot/manifest.json", manifest)
    upstream = {name: sha(checkout / name) for name in git(checkout, "ls-files").splitlines()}
    # The complete tree hash plus file hashes covers unchanged upstream modules.
    write(output / "upstream_files.json", upstream)
    toolkit = ROOT / config["environment"]["cuda_toolkit"]
    shutil.copy2(toolkit / "routescale_patch.json", output / "toolchain_patch.json")
    write(output / "provenance.json", dict(config=config, config_sha256=sha(args.config), series=series,
        patched_commit=git(checkout, "rev-parse", "HEAD"), patched_tree=git(checkout, "rev-parse", "HEAD^{tree}"),
        source_sha256=manifest, python=sys.version,
        model_files_sha256={p.name: sha(p) for p in (ROOT / config["model"]["path"]).iterdir() if p.is_file()},
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        server_command=command(config), gpu_before=memory()))
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + ":" + str(toolkit / "bin") + ":" + os.environ["PATH"],
        CUDA_HOME=str(toolkit), CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", MAX_JOBS="4",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
        MINISGL_DISABLE_OVERLAP_SCHEDULING="1", ROUTESCALE_I1_EVENTS=str(runtime),
        ROUTESCALE_MINISGL_CHECKOUT=str(checkout),
        TVM_FFI_CACHE_DIR=str(ROOT / "runs/cache/i_minisglang/tvm"),
        FLASHINFER_WORKSPACE_BASE=str(ROOT / "runs/cache/i_minisglang/flashinfer"))
    record = dict(stage="I1", attempt=args.attempt, scenario=args.scenario, status="fail",
        evidence_class="functional_not_performance", runtime_dir=str(runtime.relative_to(ROOT)),
        start_wall_time_ns=time.time_ns(),
        ipc_semaphores_before=sorted(p.name for p in Path("/dev/shm").glob("sem.*")))
    process, log, tracked = None, None, []
    try:
        import httpx
        import psutil
        started = time.monotonic()
        log = (runtime / "server.log").open("w")
        process = subprocess.Popen(command(config), cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        record["server_pid"] = process.pid
        base = f"http://{config['server']['host']}:{config['server']['port']}"
        with httpx.Client(timeout=1, trust_env=False) as http:
            while time.monotonic() - started < config["budgets"]["startup_s"]:
                if process.poll() is not None:
                    raise RuntimeError(f"Service exited during startup: {process.returncode}")
                try:
                    if http.get(base + "/ready").status_code == 200 and http.get(base + "/debug/i1/frontend").json()["pid"] == process.pid:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Startup deadline exceeded")
        record["startup_seconds"] = time.monotonic() - started
        tracked = psutil.Process(process.pid).children(recursive=True)
        record["initial_children"] = [dict(pid=p.pid, name=p.name(), created=p.create_time()) for p in tracked]
        record["gpu_serving"] = memory()
        records = asyncio.run(exercise(config, runtime, output, args.scenario, process))
        record["client_terminal_counts"] = dict(Counter(r["state"] for r in records))
        record["status"] = "scenario_exercised_pending_independent_audit"
    except (Exception, KeyboardInterrupt) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        (runtime / "runner_error.log").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        if process is not None:
            import psutil
            started = time.monotonic()
            scenario_path = output / "scenario.json"
            active_shutdown = args.scenario == "shutdown_active" and scenario_path.exists()
            if active_shutdown:
                started = json.loads(scenario_path.read_text())["monotonic_ns"] / 1e9
            if process.poll() is None and not active_shutdown:
                process.send_signal(signal.SIGTERM)  # parent owns orderly worker shutdown
            forced = False
            try:
                process.wait(timeout=max(0.1, config["budgets"]["shutdown_s"] - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                forced = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            alive = living_children(tracked)
            if alive:
                _, alive = psutil.wait_procs(alive, timeout=0.5)
            for child in alive:
                forced = True
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            record["shutdown"] = dict(seconds=time.monotonic() - started, forced_cleanup=forced,
                returncode=process.returncode, tracked_pids=[process.pid] + [p.pid for p in tracked],
                surviving_pids=[p.pid for p in living_children(tracked)])
        if log is not None:
            log.close()
        record["gpu_after"] = memory()
        record["ipc_semaphores_after"] = sorted(p.name for p in Path("/dev/shm").glob("sem.*"))
        write(output / "server_events.json", events(runtime))
        record["raw_log_sha256"] = {p.name: sha(p) for p in runtime.iterdir() if p.is_file()}
        write(output / "run.json", record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    return 0 if record["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())

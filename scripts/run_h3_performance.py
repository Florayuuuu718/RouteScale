"""Run H3 preparation, new numerical gate, pilot and 36 serial GPU processes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_performance import DEFAULT_CONFIG, SOURCES, VARIANTS, read_config, now
from scripts.summarize_h3_performance import summarize_traces, validate_run


def formal_jobs(config):
    for run in range(1, config["independent_runs"] + 1):
        order = VARIANTS[(run - 1) % 3:] + VARIANTS[:(run - 1) % 3]
        for case in config["cases"]:
            for variant in order:
                yield case["case_id"], variant, f"run{run}"


def pilot_gate(config_path):
    # The runner uses the training interpreter. Pilot reports bind GPU provenance;
    # compare that exact provenance across all six independent children.
    config = read_config(config_path)
    directory = ROOT / config["results_dir"]
    output = directory / "pilot_gate.json"
    if output.exists():
        raise ValueError("Preserve existing pilot qualification")
    workload = json.loads((ROOT / config["workload_path"]).read_text())["cases"]
    cases = {c["case_id"]: c for c in workload}
    gate = json.loads((directory / "gate.json").read_text())
    record = dict(schema_version=1, stage="H3", status="fail", created_at_utc=now(),
        qualification_gate_sha256=digest(directory / "gate.json"), checks={}, raw_sha256={})
    for case_id in config["pilot_case_ids"]:
        for variant in VARIANTS:
            path = directory / "pilot" / case_id / variant / "run1.json"
            run = json.loads(path.read_text())
            validate_run(run, config, cases[case_id], gate["provenance"], record["qualification_gate_sha256"], pilot=True)
            metrics = {scope: summarize_traces(m["traces"], scope) for scope, m in run["measurements"].items()}
            record["checks"][f"{case_id}/{variant}"] = dict(metrics=metrics,
                stable=all(max(m["wall_cv"], m["measurement_cv"]) <= config["pilot_max_coefficient_of_variation"] for m in metrics.values()))
            record["raw_sha256"][str(path.relative_to(directory))] = digest(path)
    if all(check["stable"] for check in record["checks"].values()):
        record["status"] = "pass"
    write_result(record, output)
    if record["status"] != "pass":
        raise ValueError("Pilot instability: preserve this attempt; formal matrix not started")
    print("H3 pilot gate: pass", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    jobs = list(formal_jobs(config))
    if args.dry_run:
        print(f"H3: {1 + len(config.get('additional_reference_policies', []))} reference + 3 qualification processes; {len(config['pilot_case_ids']) * 3} pilot; {len(jobs)} formal")
        print(json.dumps(jobs[:6], indent=2))
        return
    directory = ROOT / config["results_dir"]
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Fresh result directory required; never overwrite a previous attempt")
    (ROOT / "runs/tmp").mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_snapshot = {}
    config_name = str(args.config.relative_to(ROOT)) if args.config.is_absolute() else str(args.config)
    for name in dict.fromkeys((*SOURCES, config.get("protocol", "docs/h3_performance_protocol.md"), config_name)):
        target = directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        source_snapshot[name] = digest(target)
    write_result(dict(schema_version=1, source_sha256=source_snapshot), directory / "source_snapshot/manifest.json")
    log_dir = ROOT / "runs/logs/h_nanovllm" / directory.name
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "runs/cache" / directory.name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", TORCHINDUCTOR_CACHE_DIR=str(cache_dir / "inductor"),
        TRITON_CACHE_DIR=str(cache_dir / "triton"))
    python = str(ROOT / "runs/envs/nano-vllm/bin/python")
    base = [python, "-m", "serving.h3_performance"]
    common = ["--config", str(args.config)]
    steps = [("prepare", [str(ROOT / ".venv/bin/python"), "-m", "serving.h3_performance", "prepare", *common]),
        ("reference", [*base, "reference", *common])]
    steps += [(f"reference_{policy}", [*base, "reference", *common, "--reference-policy", policy])
        for policy in config.get("additional_reference_policies", [])]
    steps += [(f"qualify_{v}", [*base, "qualify", *common, "--variant", v]) for v in VARIANTS]
    steps += [("numerical_gate", [*base, "compare", *common])]
    steps += [(f"pilot_{case}_{v}", [*base, "pilot", *common, "--case-id", case, "--variant", v])
        for case in config["pilot_case_ids"] for v in VARIANTS]
    steps += [("pilot_gate", None)]
    steps += [(f"formal_{case}_{v}_{run}", [*base, "benchmark", *common, "--case-id", case, "--variant", v, "--run-id", run])
        for case, v, run in jobs]
    steps += [("independent_audit", [python, "scripts/audit_h3_performance.py", *common])]
    old_results = {str(p.relative_to(ROOT)): digest(p)
        for parent in (ROOT / "results/g_serving", ROOT / "results/h_nanovllm")
        for p in parent.rglob("*.json") if not p.is_relative_to(directory)}
    write_result(dict(schema_version=1, historical_json_sha256=old_results), directory / "historical_baseline.json")
    journal = dict(schema_version=1, stage="H3", status="running", started_at_utc=now(), pid=os.getpid(),
        config_sha256=digest(args.config), completed_steps=[], planned_formal_processes=len(jobs), completed_formal_processes=0)
    try:
        for name, command in steps:
            journal.update(current_step=name, updated_at_utc=now())
            write_result(journal, directory / "run.json")
            print(f"H3 performance: {name}", flush=True)
            if command is None:
                pilot_gate(args.config)
            else:
                with (log_dir / f"{name}.log").open("x") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        check=True, timeout=config["subprocess_timeout_seconds"])
            journal["completed_steps"].append(name)
            if name.startswith("formal_"):
                journal["completed_formal_processes"] += 1
        if any(digest(ROOT / name) != checksum for name, checksum in old_results.items()):
            raise ValueError("Historical G/H3 evidence changed")
        journal.update(status="pass", current_step="complete", historical_result_hashes_unchanged=True)
    except BaseException:
        journal.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        journal["updated_at_utc"] = now()
        write_result(journal, directory / "run.json")


if __name__ == "__main__":
    main()

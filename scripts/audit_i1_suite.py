"""Combine four independently audited real-GPU scenarios; never infer unrun modes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_suite(directories, cpu_evidence):
    scenarios, checks, digests = {}, {}, {}
    trees, configs, models = set(), set(), set()
    for directory in directories:
        read = lambda name: json.loads((directory / name).read_text())
        stored = read("audit.json")
        recomputed = runpy.run_path(str(directory / "audit_source.py"))["audit"](directory)
        scenario = stored["scenario"]
        checks[f"recomputed_{directory.name}"] = stored == recomputed and stored["lifecycle_complete"]
        if scenario in scenarios:
            raise ValueError("Exactly one accepted run per scenario required")
        run, prov, events = read("run.json"), read("provenance.json"), read("server_events.json")
        trees.add(prov["patched_tree"])
        configs.add(prov["config_sha256"])
        models.add(json.dumps(prov["model_files_sha256"], sort_keys=True))
        stopped = [row for row in events if row["event"] == "workers_stopped"][-1]
        workers = {str(p["pid"]) for p in stopped["workers"]}
        observed = {str(row["pid"]) for row in events if row["event"] in {"engine_ready", "tokenized"}}
        initial = {str(child["pid"]) for child in run["initial_children"]}
        target = str(read("scenario.json").get("target_pid")) if scenario.endswith("_fault") else None
        checks[f"worker_exitcodes_{directory.name}"] = workers == observed <= initial and len(workers) == 2 and all(
            stopped["exitcodes"][pid] == (-15 if pid == target else 0) for pid in workers)
        if scenario != "healthy":
            checks[f"ipc_released_{directory.name}"] = set(run["ipc_semaphores_after"]) <= set(run["ipc_semaphores_before"])
        scenarios[scenario] = dict(directory=str(directory), check_count=len(stored["checks"]),
            client_terminal_counts=stored["client_terminal_counts"], shutdown_s=run["shutdown"]["seconds"],
            worker_stop_s=stopped["seconds"], forced_cleanup=run["shutdown"]["forced_cleanup"])
        digests[str(directory / "audit.json")] = sha(directory / "audit.json")
        digests[str(directory / "audit_source.py")] = sha(directory / "audit_source.py")
    checks["all_four_scenarios"] = set(scenarios) == {"healthy", "scheduler_fault", "tokenizer_fault", "shutdown_active"}
    checks["same_patch_tree_config_and_model"] = len(trees) == len(configs) == len(models) == 1
    cpu = json.loads(cpu_evidence.read_text())
    checks["cpu_and_tamper_controls_passed"] = all(row["returncode"] == 0 for row in cpu["runs"]) and cpu["total_tests"] == sum(row["tests"] for row in cpu["runs"])
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "serving/online_client.py").is_file())
    checks["cpu_evidence_matches_frozen_files"] = all(sha(root / name) == digest for name, digest in cpu["source_sha256"].items())
    checks["cpu_test_logs_match_hashes"] = all(sha(root / row["output"]) == row["sha256"] for row in cpu["runs"])
    complete = all(checks.values())
    return dict(stage="I1", lifecycle_complete=complete, status="accepted" if complete else "incomplete",
        scope="single RTX 5060 Laptop, Qwen3-0.6B BF16, TP1, shared tokenizer, overlap/graph off",
        patched_tree=next(iter(trees)), checks=checks, scenarios=scenarios,
        cpu_tests=cpu["total_tests"], evidence_sha256=digests, cpu_evidence_sha256=sha(cpu_evidence),
        audit_source_sha256=sha(Path(__file__)),
        boundaries=["finite logits, not reference numerical equivalence", "no performance claim",
            "external request ID is correlation only", "TP2/4, overlap, separate tokenizer and restart recovery unaccepted",
            "request-count admission, not HTTP byte/connection/backlog bounds"],
        paid_resources_used=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs=4, required=True)
    parser.add_argument("--cpu-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit_suite(args.directories, args.cpu_evidence)
    if args.check_only:
        if json.loads(args.output.read_text()) != result:
            raise ValueError("Suite audit differs from independent recomputation")
    else:
        if args.output.exists():
            raise ValueError("Preserve previous suite audit")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "lifecycle_complete", "cpu_tests")}))
    raise SystemExit(0 if result["lifecycle_complete"] else 1)

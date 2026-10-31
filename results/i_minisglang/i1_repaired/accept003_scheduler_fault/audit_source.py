"""Recompute acceptance from frozen client, scheduler, detokenizer and process evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "serving/online_client.py").is_file())
sys.path.insert(0, str(ROOT))
from serving.online_client import canonical_hash, TERMINAL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    run, prov = read("run.json"), read("provenance.json")
    checks = {}
    def check(name, passed, **details):
        checks[name] = dict(status="pass" if passed else "fail", **details)

    config = prov["config"]
    manifest = read("source_snapshot/manifest.json")
    check("frozen_sources", manifest == prov["source_sha256"] and all(
        sha(directory / "source_snapshot" / name) == digest for name, digest in manifest.items()))
    check("frozen_config", sha(directory / "source_snapshot/config/serving/i1_repaired.json") == prov["config_sha256"])
    series = read("source_snapshot/patches/minisgl-i1/series.json")
    check("replayable_patch_identity", series == prov["series"] and series["tree"] == prov["patched_tree"] and
        series["base_commit"] == config["upstream"]["commit"] and len(series["patch_sha256"]) == 7 and
        all(sha(directory / "source_snapshot/patches/minisgl-i1" / name) == digest for name, digest in series["patch_sha256"].items()))
    raw = ROOT / run["runtime_dir"]
    check("raw_logs", {p.name for p in raw.iterdir() if p.is_file()} == set(run["raw_log_sha256"]) and
          all((raw / name).is_file() and sha(raw / name) == digest for name, digest in run["raw_log_sha256"].items()))
    check("runner_completed", run["status"] == "scenario_exercised_pending_independent_audit")
    check("frozen_single_gpu_scope", config["server"]["tensor_parallel_size"] == 1 and
          config["server"]["num_tokenizer"] == 0 and config["server"]["cuda_graph_max_bs"] == 0 and
          config["server"]["overlap_scheduling"] is False)
    if run["status"] == "fail":
        return dict(stage="I1", scenario=run["scenario"], status="execution_failed", lifecycle_complete=False, checks=checks)
    requests, workload, rows = read("requests.json"), read("workload.json"), read("server_events.json")
    raw_rows = sorted([json.loads(line) for p in raw.glob("events.*.jsonl") for line in p.read_text().splitlines()], key=lambda r: r["monotonic_ns"])
    check("server_events_match_raw", rows == raw_rows)
    client_events = defaultdict(list)
    for line in (raw / "client.jsonl").read_text().splitlines():
        event = json.loads(line)
        client_events[event["request_id"]].append(event)
    check("client_events_match_raw", set(client_events) == {r["request_id"] for r in requests} and
          all(r["events"] == client_events[r["request_id"]] for r in requests))
    by_name = {r["request_id"]: r for r in requests}
    specs = {w["request_id"]: w for w in workload["requests"]}
    check("every_offered_request_has_one_terminal", len(by_name) == len(requests) == len(specs) == len(workload["requests"]) and
        set(by_name) == set(specs) and all(r["state"] in TERMINAL and
        sum(e["event"] in TERMINAL for e in r["events"]) == 1 for r in requests))
    check("frozen_payload_and_prompt_hashes", all(r["payload_sha256"] == specs[r["request_id"]]["payload_sha256"] ==
          canonical_hash(specs[r["request_id"]]["payload"]) and specs[r["request_id"]]["input_ids_sha256"] ==
          canonical_hash(specs[r["request_id"]]["input_ids"]) for r in requests))
    external, local = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["event"] == "frontend_new_user":
            external[row["external_id"]].append(row["uid"])
        if "uid" in row:
            local[row["uid"]].append(row)
    details = {}
    for record in requests:
        rid = record["request_id"]
        spec = specs[rid]
        header = spec["options"].get("external_id", rid)
        uid = external[header].pop(0) if external[header] else None
        uid_rows = local[uid] if uid is not None else []
        tokens = [r for r in uid_rows if r["event"] == "scheduler_token"]
        tokenized = [r for r in uid_rows if r["event"] == "tokenized"]
        budget = min(spec["payload"]["max_tokens"], config["server"]["max_seq_len_override"] - len(spec["input_ids"])) if isinstance(spec["payload"]["max_tokens"], int) else None
        expected = ("stop" if tokens[-1]["token_id"] == workload["eos_token_id"] and not spec["payload"]["ignore_eos"]
                    else "length" if len(tokens) == budget else "unfinished") if tokens else None
        terminal = next(e for e in record["events"] if e["event"] in TERMINAL)
        details[rid] = dict(uid=uid, state=record["state"], output_tokens=len(tokens),
            backend_terminal_tokens=sum(t["finished"] for t in tokens), backend_reason=expected,
            wire_reason=tokens[-1].get("finish_reason") if tokens else None,
            client_reason=record["finish_reason"], http_status=record["http_status"],
            free_events=sum(r["event"] == "request_free" for r in uid_rows),
            input_ids_match=tokenized[0]["input_ids"] == spec["input_ids"] if tokenized else None,
            terminal_ns=terminal["monotonic_ns"])
        if record["response_id"]:
            check(f"response_uid_{rid}", re.fullmatch(r"(?:chat)?cmpl-\d+", record["response_id"]) is not None and
                  int(record["response_id"].rsplit("-", 1)[-1]) == uid)
    check("tokenizer_matches_prompts", all(d["input_ids_match"] is True for rid, d in details.items()
          if d["uid"] is not None and by_name[rid]["http_status"] != 503))
    finite = [r for r in rows if r["event"] == "logits_check"]
    check("real_kernel_and_finite_logits", bool(finite) and all(r["finite"] for r in finite) and
        any(r["event"] == "engine_ready" and r["backend"] == "FlashInferBackend" and r["dtype"] == "torch.bfloat16" for r in rows),
        checked_batches=len(finite))
    completed = [rid for rid, d in details.items() if d["state"] == "finished"]
    check("completed_requests_release_once_and_report_real_reason", bool(completed) and all(
        details[rid]["free_events"] == 1 and details[rid]["backend_terminal_tokens"] == 1 and
        details[rid]["backend_reason"] in {"stop", "length"} and
        details[rid]["client_reason"] == details[rid]["wire_reason"] == details[rid]["backend_reason"] for rid in completed))
    check("short_request_eos", details["eos_short"]["state"] == "finished" and details["eos_short"]["backend_reason"] == "stop")
    admission = [r for r in rows if r["event"] in {"frontend_new_user", "admission_release", "frontend_rejected"}]
    limit = config["server"]["max_total_requests"]
    check("admission_always_bounded", bool(admission) and all(len(r["admitted_uids"]) <= limit and
          len(r["admitted_uids"]) == len(set(r["admitted_uids"])) for r in admission), limit=limit,
          peak=max(len(r["admitted_uids"]) for r in admission))
    releases = [r for r in rows if r["event"] == "admission_release"]
    check("admission_waits_for_frontend_and_backend", bool(releases) and all(
        r["uid"] not in r["ack_uids"] and any(d["monotonic_ns"] < r["monotonic_ns"] and
            (d["event"] == "detokenizer_end" or (d["event"] == "detokenizer_state" and r["uid"] in d["uids"])) and
            r["uid"] not in d["active_uids"] for d in rows if d["event"] in {"detokenizer_state", "detokenizer_end"})
        for r in releases))
    snapshots = read("frontend_snapshots.json")
    settlements = read("settlements.json")
    settled_ns = {s["snapshot_ns"] for s in settlements}
    normal_snaps = [s for s in snapshots if s["monotonic_ns"] in settled_ns]
    check("settled_frontend_refs_empty", bool(normal_snaps) and all(
          not s[k] for s in normal_snaps for k in ("ack_uids", "event_uids", "deadline_uids", "admitted_uids", "aborting_uids", "backend_done_uids")))
    # Inspect the rows referenced by every settlement, rather than trust runner booleans.
    indexed = {r["monotonic_ns"]: r for r in rows}
    resources = []
    for settled in settlements:
        idle, detok = indexed[settled["idle_ns"]], indexed[settled["detokenizer_ns"]]
        resources.append(idle["event"] == "scheduler_idle" and
            idle["available_tables"] == config["server"]["max_running_requests"] and
            idle["prefix_total_tokens"] == idle["prefix_evictable_tokens"] and
            idle["free_pages"] + idle["prefix_total_tokens"] == idle["num_pages"] and
            not idle["pending_uids"] and not idle["running_uids"] and not detok["active_uids"] and
            (settled["ended_ns"] - settled["started_ns"]) / 1e9 <= config["budgets"]["cleanup_s"])
    check("settled_table_kv_and_detokenizer_refs", bool(resources) and all(resources), settlements=len(resources))

    if run["scenario"] == "healthy":
        expected_finished = {"eos_short", "max_tokens_32", "concurrent_0", "concurrent_1", "concurrent_2",
            "queue_blocker", "nonstream_eos", "nonstream_length", "post_cancel_probe", "context_last_slot",
            "repeated_header_1", "repeated_header_2", "post_overload_probe"}
        check("normal_matrix_finished", expected_finished <= set(completed))
        check("stream_and_nonstream_length_budgets", details["max_tokens_32"]["output_tokens"] == 32 and
            details["nonstream_length"]["output_tokens"] == 8 and details["context_last_slot"]["output_tokens"] == 1 and
            details["nonstream_eos"]["backend_reason"] == "stop")
        check("invalid_parameters_rejected_before_uid", all(details[rid]["state"] == "rejected" and
            details[rid]["http_status"] == 422 and details[rid]["uid"] is None for rid in ("invalid_type", "zero_budget")))
        check("context_overflow_and_equal_explicitly_rejected", all(details[rid]["state"] == "rejected" and
            details[rid]["http_status"] == 400 and details[rid]["output_tokens"] == 0 and
            any(r["event"] == "scheduler_end" and r["status_code"] == 400 for r in local[details[rid]["uid"]])
            for rid in ("context_overflow", "context_equal")))
        cancellation = {}
        for rid in ("queued_cancel", "decode_disconnect", "decode_deadline", "nonstream_cancel"):
            d = details[rid]
            uid, terminal_ns = d["uid"], d["terminal_ns"]
            aborts = [r for r in local[uid] if r["event"] == "scheduler_receive" and r["kind"] == "AbortBackendMsg"]
            ends = [r for r in local[uid] if r["event"] == "detokenizer_end"]
            releases = [r for r in local[uid] if r["event"] == "admission_release"]
            after = [r for r in rows if aborts and r["monotonic_ns"] > aborts[0]["monotonic_ns"] and
                ((r["event"] == "batch_prepare" and uid in r["uids"]) or (r["event"] == "scheduler_token" and r["uid"] == uid))]
            cancellation[rid] = dict(abort_count=len(aborts), detokenizer_end_count=len(ends),
                delay_s=(ends[-1]["monotonic_ns"] - terminal_ns) / 1e9 if ends else None,
                after_abort_work=len(after), release_after_detok=bool(ends and releases) and
                releases[-1]["monotonic_ns"] > ends[-1]["monotonic_ns"])
        check("cancellation_stops_backend_cleans_detok_then_releases_admission", all(details[rid]["state"] == "cancelled" and
            c["abort_count"] == c["detokenizer_end_count"] == 1 and c["delay_s"] <= config["budgets"]["cleanup_s"] and
            c["after_abort_work"] == 0 and c["release_after_detok"] for rid, c in cancellation.items()), requests=cancellation)
        uid = details["queued_cancel"]["uid"]
        check("queued_cancel_never_executes", details["queued_cancel"]["output_tokens"] == 0 and
              not any(r["event"] == "batch_prepare" and uid in r["uids"] for r in rows))
        check("active_cancellations_free_once", all(details[rid]["output_tokens"] > 0 and details[rid]["free_events"] == 1
            for rid in ("decode_disconnect", "decode_deadline", "nonstream_cancel")))
        overloaded = [d for rid, d in details.items() if rid.startswith("overload_")]
        check("overload_rejects_and_recovers", len(overloaded) == 12 and
            sum(d["state"] == "finished" for d in overloaded) == limit and
            all(d["state"] == "finished" or (d["state"] == "rejected" and d["http_status"] == 429 and d["uid"] is None) for d in overloaded) and
            details["post_overload_probe"]["state"] == "finished")
        check("external_id_contract_is_fresh_uid", details["repeated_header_1"]["uid"] != details["repeated_header_2"]["uid"],
              boundary="External X-Request-ID is a correlation header; no idempotence guarantee")
    else:
        fault = read("scenario.json")
        client_budget = config["budgets"]["shutdown_s" if run["scenario"] == "shutdown_active" else "worker_failure_s"]
        check("inflight_and_queued_fail_within_budget", all(details[rid]["state"] == "failed" and
            0 <= (details[rid]["terminal_ns"] - fault["monotonic_ns"]) / 1e9 <= client_budget
            for rid in ("fault_inflight", "fault_queued")), budget_s=client_budget)
        check("fault_injected_during_real_decode", details["fault_inflight"]["output_tokens"] >= 4 and
              details["fault_queued"]["output_tokens"] == 0)
        if run["scenario"] != "shutdown_active":
            check("worker_fault_readiness_and_admission_503", fault["health_status"] == 503 and
                details["post_fault_admission"]["http_status"] == 503 and details["post_fault_admission"]["uid"] is None)
            check("inflight_failure_is_explicit_sse_error", any(e["event"] == "server_error" and e["error"]["code"] == 503
                for e in by_name["fault_inflight"]["events"]))
            final_fault = snapshots[-1]
            check("worker_fault_frontend_state_released", final_fault["label"] == "after_fault" and
                all(not final_fault[k] for k in ("ack_uids", "event_uids", "deadline_uids", "admitted_uids", "aborting_uids", "backend_done_uids")) and
                (fault["frontend_cleanup_ns"] - fault["monotonic_ns"]) / 1e9 <= config["budgets"]["cleanup_s"])

    stopped = [r for r in rows if r["event"] == "workers_stopped"]
    check("workers_stopped_without_internal_escalation", bool(stopped) and all(
        not r["escalated"] and not r["surviving_pids"] for r in stopped), reports=stopped)
    shutdown = run["shutdown"]
    check("bounded_shutdown_without_runner_force", not shutdown["forced_cleanup"] and not shutdown["surviving_pids"] and
        shutdown["seconds"] <= config["budgets"]["shutdown_s"], shutdown=shutdown)
    check("gpu_memory_returned_to_idle", int(run["gpu_after"].split(",")[2].strip().split()[0]) == 0,
          gpu_after=run["gpu_after"])
    complete = all(c["status"] == "pass" for c in checks.values())
    evidence = {p.name: sha(p) for p in sorted(directory.glob("*.json")) if p.name != "audit.json"}
    return dict(stage="I1", scenario=run["scenario"], status="scenario_pass" if complete else "acceptance_failed",
        lifecycle_complete=complete, performance_claims_allowed=False,
        scope="TP1, shared tokenizer, overlap/graph off; finite logits are not reference equivalence",
        audit_source_sha256=sha(Path(__file__)),
        checks=checks, failed_checks=[name for name, c in checks.items() if c["status"] == "fail"],
        requests=details, client_terminal_counts=dict(Counter(r["state"] for r in requests)), evidence_sha256=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.directory)
    path = args.directory / "audit.json"
    if args.check_only:
        if json.loads(path.read_text()) != result:
            raise ValueError("Stored audit differs from recomputed evidence")
    else:
        if path.exists():
            raise ValueError("Preserve old audit; use --check-only")
        (args.directory / "audit_source.py").write_bytes(Path(__file__).read_bytes())
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result.get(key) for key in ("status", "scenario", "lifecycle_complete", "failed_checks")}))
    return 0 if result["lifecycle_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

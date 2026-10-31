"""Audit I1 client/server evidence independently; distinguish reproduction from acceptance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.online_client import canonical_hash, TERMINAL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    read = lambda name: json.loads((directory / name).read_text())
    run, provenance = read("run.json"), read("provenance.json")
    checks = {}
    def check(name, passed, **details):
        checks[name] = dict(status="pass" if passed else "fail", **details)

    manifest = read("source_snapshot/manifest.json")
    check("frozen_sources", all(sha(directory / "source_snapshot" / n) == digest for n, digest in manifest.items()))
    check("frozen_config", sha(directory / "source_snapshot/config/serving/i1_minisglang.json") ==
          provenance["config_sha256"])
    raw = ROOT / run["runtime_dir"]
    check("raw_logs", all((raw / n).is_file() and sha(raw / n) == digest
                          for n, digest in run["raw_log_sha256"].items()))
    if run["status"] == "fail" or not (directory / "requests.json").exists():
        return dict(stage="I1", status="startup_or_warmup_failed", checks=checks, lifecycle_complete=False)

    requests, workload, rows = read("requests.json"), read("workload.json"), read("server_events.json")
    snapshots = read("frontend_snapshots.json")
    raw_rows = [json.loads(line) for p in raw.glob("events.*.jsonl") for line in p.read_text().splitlines()]
    check("server_events_match_raw_logs", rows == sorted(raw_rows, key=lambda r: r["monotonic_ns"]))
    client_events = defaultdict(list)
    for line in (raw / "client.jsonl").read_text().splitlines():
        event = json.loads(line)
        client_events[event["request_id"]].append(event)
    check("client_events_match_raw_logs", all(r["events"] == client_events[r["request_id"]] for r in requests))
    by_name = {r["request_id"]: r for r in requests}
    check("one_client_terminal_per_request", len(by_name) == len(requests) and all(
        r["state"] in TERMINAL and sum(e["event"] in TERMINAL for e in r["events"]) == 1 for r in requests))
    check("client_payload_hashes", all(r["payload_sha256"] == w["payload_sha256"] == canonical_hash(w["payload"])
        for r, w in zip(requests, workload["requests"], strict=True)))
    by_uid = defaultdict(list)
    for row in rows:
        if "uid" in row:
            by_uid[row["uid"]].append(row)
    external = defaultdict(list)
    for row in rows:
        if row["event"] == "frontend_new_user":
            external[row["external_id"]].append(row["uid"])
    assignment, details = {}, {}
    eos = workload["eos_token_id"]
    for record, w in zip(requests, workload["requests"], strict=True):
        rid = record["request_id"]
        header = w["options"].get("external_id", rid)
        uid = external[header].pop(0) if external[header] else None
        assignment[rid] = uid
        tokens = [r for r in by_uid[uid] if r["event"] == "scheduler_token"] if uid is not None else []
        tokenized = [r for r in by_uid[uid] if r["event"] == "tokenized"] if uid is not None else []
        model_end = ("eos" if tokens[-1]["token_id"] == eos and not w["payload"].get("ignore_eos")
                     else "length" if len(tokens) == w["payload"]["max_tokens"] else "other") if tokens else None
        details[rid] = dict(uid=uid, client_terminal=record["state"], output_tokens=len(tokens),
            backend_finished=tokens[-1]["finished"] if tokens else False,
            backend_end_reason=model_end, reported_finish_reason=record["finish_reason"],
            input_ids_match=tokenized[0]["input_ids"] == w["input_ids"] if tokenized else None,
            request_free_events=sum(r["event"] == "request_free" for r in by_uid[uid]) if uid is not None else 0)
        if record["response_id"] and uid is not None:
            if re.fullmatch(r"cmpl-\d+", record["response_id"]) is None or int(record["response_id"].split("-")[-1]) != uid:
                raise ValueError("Response UID and observed external header differ")

    check("tokenizer_matches_frozen_prompts", all(d["input_ids_match"] for d in details.values()
          if d["input_ids_match"] is not None))
    finite = [r for r in rows if r["event"] == "logits_check"]
    check("real_kernel_finite_logits", bool(finite) and all(r["finite"] for r in finite),
          checked_batches=len(finite))
    normal = ["eos_short", "max_tokens_32", "concurrent_0", "concurrent_1", "concurrent_2",
              "queue_blocker", "post_cancel_probe", "repeated_header_1", "repeated_header_2"]
    check("normal_stream_completion", all(by_name.get(n, {}).get("state") == "finished" and
          details[n]["backend_finished"] and details[n]["request_free_events"] == 1 for n in normal))
    check("eos_respected", details["eos_short"]["backend_end_reason"] == "eos")
    check("output_budget_respected", details["max_tokens_32"]["output_tokens"] == 32 and
          details["max_tokens_32"]["backend_end_reason"] == "length")
    check("finish_reason_matches_backend", all(details[n]["reported_finish_reason"] ==
          ("stop" if details[n]["backend_end_reason"] == "eos" else "length") for n in normal),
          note="Upstream may always report stop, including length exhaustion")
    cleanup_s = provenance["config"]["budgets"]["cleanup_s"]
    cancellation = {}
    for rid in ("queued_cancel", "decode_disconnect", "decode_deadline"):
        uid = assignment[rid]
        local = by_uid[uid]
        aborts = [r for r in local if r["event"] == "scheduler_receive" and r["kind"] == "AbortBackendMsg"]
        terminal = next(e for e in by_name[rid]["events"] if e["event"] in TERMINAL)
        after_abort = [r for r in rows if r["event"] == "batch_prepare" and uid in r["uids"] and
                       aborts and r["monotonic_ns"] > aborts[-1]["monotonic_ns"]]
        delay = (aborts[0]["monotonic_ns"] - terminal["monotonic_ns"]) / 1e9 if aborts else None
        cancellation[rid] = dict(uid=uid, abort_received=bool(aborts), delay_seconds=delay,
            further_batches_after_abort=len(after_abort), output_tokens=details[rid]["output_tokens"])
    check("cancellation_reaches_scheduler", all(c["abort_received"] and c["delay_seconds"] <= cleanup_s and
          c["further_batches_after_abort"] == 0 for c in cancellation.values()), requests=cancellation)
    check("queued_cancel_never_executes", cancellation["queued_cancel"]["output_tokens"] == 0 and
          not any(r["event"] == "batch_prepare" and assignment["queued_cancel"] in r["uids"] for r in rows))
    idle = [r for r in rows if r["event"] == "scheduler_idle"]
    last_idle = idle[-1] if idle else None
    check("idle_table_and_kv_refs_released", bool(last_idle) and last_idle["available_tables"] ==
          provenance["config"]["server"]["max_running_requests"] and
          last_idle["prefix_total_tokens"] == last_idle["prefix_evictable_tokens"] and
          last_idle["free_pages"] + last_idle["prefix_total_tokens"] == last_idle["num_pages"] and
          not last_idle["pending_uids"] and not last_idle["running_uids"], last_idle=last_idle)
    before_fault = next(s for s in snapshots if s["label"] == "after_repeated_external_id")
    check("frontend_refs_released", not before_fault["ack_uids"] and not before_fault["event_uids"],
          snapshot=before_fault)
    detokenizer = [r for r in rows if r["event"] == "detokenizer_state"]
    last_detok = detokenizer[-1] if detokenizer else None
    check("detokenizer_refs_released", bool(last_detok) and not last_detok["active_uids"], snapshot=last_detok)
    check("invalid_type_rejected", by_name["invalid_type"]["state"] == "rejected" and
          assignment["invalid_type"] is None)
    check("over_context_rejected_with_error", by_name["context_overflow"]["state"] == "rejected")
    fault = read("worker_failure.json")
    check("worker_fault_client_bounded", by_name["worker_failure"]["state"] in {"failed", "cancelled"})
    check("worker_fault_health_propagates", fault["frontend_health_status"] != 200, fault=fault)
    checks["bounded_admission"] = dict(status="unavailable",
        reason="max_running_requests bounds execution slots; upstream pending list has no configured admission limit")
    checks["external_id_idempotence"] = dict(status="unavailable", observed_uids=[assignment["repeated_header_1"],
        assignment["repeated_header_2"]], reason="No native external-ID deduplication contract")
    shutdown = run["shutdown"]
    check("all_processes_reaped", not shutdown["surviving_pids"])
    check("shutdown_without_forced_cleanup", not shutdown["forced_cleanup"])
    complete = all(c["status"] == "pass" for c in checks.values())
    return dict(stage="I1", status="lifecycle_pass" if complete else "baseline_reproduced_with_lifecycle_gaps",
        lifecycle_complete=complete,
        performance_claims_allowed=False, multi_gpu="deferred_to_consolidated_rental_session",
        client_terminal_counts=dict(Counter(r["state"] for r in requests)), requests=details,
        checks=checks, failed_checks=[n for n, c in checks.items() if c["status"] == "fail"],
        unavailable_checks=[n for n, c in checks.items() if c["status"] == "unavailable"],
        evidence_sha256={n: sha(directory / n) for n in ("run.json", "provenance.json", "requests.json",
            "workload.json", "server_events.json", "frontend_snapshots.json", "worker_failure.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.directory)
    path = args.directory / "audit.json"
    if args.check_only:
        if json.loads(path.read_text()) != result:
            raise ValueError("Stored audit differs from independently recomputed evidence")
    else:
        if path.exists():
            raise ValueError("Do not overwrite an audit; use --check-only")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], lifecycle_complete=result["lifecycle_complete"],
                         failed_checks=result.get("failed_checks")), ensure_ascii=False))


if __name__ == "__main__":
    main()

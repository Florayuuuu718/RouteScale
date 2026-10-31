"""Read-only lineage checks; writes only a NEW KV-study verification report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h2_runtime import now
from serving.verify_h2_kv_capacity import DEFAULT_CONFIG, load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, inputs, provenance, _ = load_inputs(args.config)
    directory = ROOT / config["results_dir"]
    output = directory / "verification.json"
    if output.exists() and not args.check_only:
        raise ValueError("Preserve existing verification; use --check-only to recheck")
    audit = json.loads((directory / "audit.json").read_text())
    journal = json.loads((directory / "run.json").read_text())
    if (audit["status"] != "pass" or audit["provenance"] != provenance
            or audit["performance_execution_allowed"] or journal["status"] != "pass"
            or journal["completed_steps"] != ["static_window", "paged", "audit"]):
        raise ValueError("Incomplete or changed KV-study qualification")
    for name, checksum in audit["raw_sha256"].items():
        if digest(directory / name) != checksum:
            raise ValueError("KV-study raw report changed after its audit")
        report = json.loads((directory / name).read_text())
        for prefix in ("logits", "reference"):
            if digest(ROOT / report[f"{prefix}_path"]) != report[f"{prefix}_sha256"]:
                raise ValueError("Saved numerical tensor changed")
    historical, historical_audits = {}, {}
    for name, expected in (("h2_graph_b8", 18), ("h2_corrected_pilot", 36), ("h2_scheduling_tile512", 18)):
        old = ROOT / "results/h_nanovllm" / name
        historical_audits[name] = digest(old / "audit.json")
        hashes = json.loads((old / "audit.json").read_text())["raw_sha256"]
        if len(hashes) != expected:
            raise ValueError("Historical matrix count changed")
        for relative, checksum in hashes.items():
            if digest(old / relative) != checksum:
                raise ValueError(f"Historical raw result changed: {name}/{relative}")
        historical[name] = len(hashes)
    scheduling = inputs[0]
    failed_dir = ROOT / scheduling["previous_failed_results"]
    failed_gate = json.loads((failed_dir / "preflight.json").read_text())
    if failed_gate["status"] != "fail":
        raise ValueError("Original failed scheduling gate must stay failed")
    for policy, checksum in failed_gate["report_sha256"].items():
        if digest(failed_dir / "preflight" / f"{policy}.json") != checksum:
            raise ValueError("Original failed numerical probe changed")
    if subprocess.check_output(["git", "diff", "--", "docs/RouteScale_实施教程大纲.md"], cwd=ROOT, text=True):
        raise ValueError("Implementation outline has an unexpected diff")
    report = dict(schema_version=1, stage="H2", status="pass", created_at_utc=now(),
        evidence_class="kv_study_lineage_verification", verifier_sha256=digest(Path(__file__)),
        current_audit_sha256=digest(directory / "audit.json"), source_model_environment_match=True,
        numerical_tensor_hashes_match=True, historical_formal_raw_hashes_match=historical,
        historical_audit_sha256=historical_audits, original_failure_and_source_snapshot_match=True,
        outline_git_diff_empty=True, performance_execution_allowed=False)
    if not args.check_only:
        write_result(report, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

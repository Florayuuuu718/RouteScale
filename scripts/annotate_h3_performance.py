"""Reproduce post-hoc timing variability annotations without changing raw data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest
from serving.h3_performance import DEFAULT_CONFIG
from scripts.summarize_h3_performance import summarize_traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = ROOT / config["results_dir"]
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    audit = json.loads((directory / "audit.json").read_text())
    if audit["summary_sha256"] != digest(summary_path) or summary["provenance"]["config_sha256"] != digest(args.config):
        raise ValueError("Audited summary or configuration changed")
    cases = {c["case_id"]: c for c in config["cases"]}
    rows, requests, tokens = [], 0, 0
    for relative, checksum in audit["raw_sha256"].items():
        path = directory / relative
        if digest(path) != checksum:
            raise ValueError("Audited raw evidence changed")
        run = json.loads(path.read_text())
        for scope, measurement in run["measurements"].items():
            stats = summarize_traces(measurement["traces"], scope)
            rows.append(dict(case_id=run["case_id"], variant=run["variant"], run_id=run["run_id"],
                measurement_scope=scope, wall_cv=stats["wall_cv"], measurement_cv=stats["measurement_cv"],
                output_tokens_per_second=stats["output_tokens_per_second"]))
            tokens += sum(t["output_tokens"] for t in measurement["traces"])
            requests += len(measurement["traces"]) * cases[run["case_id"]]["batch_size"]
    quality = dict(schema_version=1, stage="H3", status="timing_variability_requires_followup",
        evidence_class="measurement_variability_annotation", source_sha256=digest(Path(__file__)),
        summary_sha256=digest(summary_path), measured_request_budgets=requests, measured_output_positions=tokens,
        processes_scopes_over_pilot_cv=[r for r in rows if max(r["wall_cv"], r["measurement_cv"]) > config["pilot_max_coefficient_of_variation"]],
        per_process_scope_statistics=rows,
        maximum_independent_process_throughput_ratio=max(r["throughput_max_min_ratio"] for r in summary["variants"]),
        limitations="Pilot qualified two predeclared cases. Other formal cases contain timing variability. "
        "These annotations retain every raw run and do not reject or replace formal data.")
    (directory / "measurement_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    with (directory / "per_process_timing.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"H3 timing annotations: {len(rows)} process/scope rows, "
          f"{len(quality['processes_scopes_over_pilot_cv'])} over the pilot CV threshold")


if __name__ == "__main__":
    main()

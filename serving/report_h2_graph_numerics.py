"""Report every Graph preflight comparison, including cases after the first failure."""

from __future__ import annotations

import json
from pathlib import Path

from serving.backends.nanovllm import ROOT, digest, write_result


def main():
    import torch
    pilot = json.loads((ROOT / "config/serving/h2_graph_pilot.json").read_text())
    directory = ROOT / pilot["results_dir"]
    report = {"schema_version": 1, "stage": "H2", "status": "fail",
        "evidence_class": "numerical_diagnostic", "gpu_execution_allowed": False,
        "reporter_sha256": digest(Path(__file__)), "cases": []}
    for case_id in pilot["correctness_case_ids"]:
        reports, logits = {}, {}
        for name in pilot["variants"]:
            row = json.loads((directory / "preflight" / case_id / f"{name}.json").read_text())
            if row["status"] != "pass" or row["evidence_class"] != "gpu_preflight":
                raise ValueError("Invalid source preflight")
            path = ROOT / row["numerical_probe"]["path"]
            if digest(path) != row["numerical_probe"]["sha256"]:
                raise ValueError("Probe tensor changed")
            reports[name], logits[name] = row, torch.load(path, weights_only=True)
        a, b = [reports[name] for name in pilot["variants"]]
        if a["provenance"] != b["provenance"] or a["workload"] != b["workload"]:
            raise ValueError("Cross-variant invariants differ")
        comparisons = []
        for phase in ("cold", "warm"):
            for step, (expected, actual) in enumerate(zip(logits["eager_no_prefix"][phase], logits["graph_no_prefix"][phase])):
                delta = actual - expected
                relative_rmse = (delta.square().mean() / expected.square().mean()).sqrt().item()
                passed = (torch.allclose(expected, actual, atol=pilot["correctness_atol"], rtol=pilot["correctness_rtol"])
                    and relative_rmse <= pilot["correctness_max_relative_rmse"])
                comparisons.append({"phase": phase, "step": step, "pass": passed,
                    "max_abs_error": delta.abs().max().item(), "relative_rmse": relative_rmse,
                    "top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()})
        report["cases"].append({"case_id": case_id, "pass": all(row["pass"] for row in comparisons),
            "comparisons": comparisons,
            "source_report_sha256": {name: digest(directory / "preflight" / case_id / f"{name}.json") for name in reports}})
    report["limitations"] = "This completes the failed four-case gate's diagnostic; it does not release that gate or claim performance for excluded cases."
    write_result(report, directory / "all_case_numerical_diagnostic.json")
    print([(row["case_id"], row["pass"]) for row in report["cases"]])


if __name__ == "__main__":
    main()

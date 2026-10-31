"""Compare forced-input GPU logits and release a provenance-bound H2 pilot gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from serving.backends.nanovllm import ROOT, digest, write_result
from serving.h2_runtime import DEFAULT_PILOT, check_capacity, load_inputs, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PILOT)
    args = parser.parse_args()
    pilot, config, _, cases = load_inputs(args.config)
    directory = ROOT / pilot["results_dir"]
    report = {"schema_version": 1, "stage": "H2", "status": "running",
        "evidence_class": "gpu_preflight_gate", "gpu_execution_allowed": False,
        "created_at_utc": now(), "comparisons": []}
    try:
        import torch
        names = list(pilot.get("variants", config["variants"]))
        case_ids = pilot.get("correctness_case_ids", [pilot["correctness_case_id"]])
        baseline, files = None, {}
        for case_id in case_ids:
            rows, tensors = {}, {}
            for name in names:
                file = directory / "preflight" / case_id / f"{name}.json"
                row = json.loads(file.read_text())
                if (row["status"] != "pass" or row["evidence_class"] != "gpu_preflight"
                        or row["case_id"] != case_id
                        or row["numerical_probe"]["forced_output_tokens"] != pilot["correctness_output_tokens"]):
                    raise ValueError(f"Invalid preflight report: {case_id}/{name}")
                expected_engine = {**pilot["engine"], **config["variants"][name], "dtype": config["dtype"]}
                if row["engine"] != expected_engine:
                    raise ValueError("Preflight engine switches or common budget changed")
                path = ROOT / row["numerical_probe"]["path"]
                if digest(path) != row["numerical_probe"]["sha256"]:
                    raise ValueError("Numerical probe tensor changed")
                rows[name], tensors[name] = row, torch.load(path, weights_only=True, map_location="cpu")
                files[f"{case_id}/{name}"] = digest(file)
            case_baseline = rows["eager_no_prefix"]
            baseline = baseline or case_baseline
            for row in rows.values():
                if row["provenance"] != baseline["provenance"] or row["workload"] != case_baseline["workload"]:
                    raise ValueError("Preflight provenance/workload differs across variants")
            for name in names:
                if name == "eager_no_prefix":
                    continue
                for phase in ("cold", "warm"):
                    reference, candidate = tensors["eager_no_prefix"][phase], tensors[name][phase]
                    if len(reference) != pilot["correctness_output_tokens"] or len(candidate) != len(reference):
                        raise ValueError("Numerical decode steps are incomplete")
                    for step, (expected, actual) in enumerate(zip(reference, candidate)):
                        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                            raise AssertionError("Nonfinite logits")
                        torch.testing.assert_close(actual, expected, atol=pilot["correctness_atol"], rtol=pilot["correctness_rtol"])
                        delta = actual - expected
                        relative_rmse = (delta.square().mean() / expected.square().mean()).sqrt().item()
                        if relative_rmse > pilot["correctness_max_relative_rmse"]:
                            raise AssertionError("Logit relative RMSE exceeds the predeclared threshold")
                        report["comparisons"].append({"case_id": case_id, "variant": name, "phase": phase, "step": step,
                            "max_abs_error": delta.abs().max().item(), "relative_rmse": relative_rmse,
                            "top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()})
        report.update(status="pass", gpu_execution_allowed=True,
            provenance=baseline["provenance"], common_engine=pilot["engine"],
            scope=pilot.get("scope", "cuda_graph_and_prefix_cache"), allowed_variants=names,
            validated_case_ids=case_ids,
            tolerances={k: v for k, v in pilot.items() if k.startswith("correctness_")},
            capacity_by_case={name: check_capacity(pilot["engine"], cases[name])
                for name in pilot.get("pilot_case_ids", config["pilot_case_ids"])},
            preflight_files=files,
            limitations="Cross-path BF16 equivalence on forced inputs; not a Transformers reference or a model-quality evaluation.")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        write_result(report, directory / "preflight.json")
    print(f"H2 preflight passed: {len(report['comparisons'])} logit comparisons", flush=True)


if __name__ == "__main__":
    main()

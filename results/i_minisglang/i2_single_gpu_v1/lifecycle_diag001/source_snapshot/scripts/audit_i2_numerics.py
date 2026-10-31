"""Independently re-open every full-vocabulary tensor and fail closed."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.i2_workloads import digest, token_hash, validate_workload


def audit_numerics(directory):
    import torch
    torch.set_num_threads(4)
    read = lambda name: json.loads((directory / name).read_text())
    meta, workload = read("provenance.json"), read("workload.json")
    config = meta["config"]
    validate_workload(workload, config)
    checks = {}
    checks["frozen_sources"] = all(digest(directory / "source_snapshot" / n) == value for n, value in meta["source_sha256"].items())
    ref_record = read("reference.json")
    path = ROOT / ref_record["reference"]["tensor_file"]
    checks["reference_hash"] = digest(path) == ref_record["reference"]["tensor_sha256"]
    reference = torch.load(path, weights_only=True, mmap=True)
    expected_ids = {r["request_id"] for case in workload["cases"] for phase in ("cold", "warm") for r in case["waves"][phase]}
    checks["reference_coverage"] = set(reference) == expected_ids and all(t.shape == (config["output_tokens"], 151936) for t in reference.values())
    policy = config["numerics"]
    rows, vectors = [], 0
    for name, variant in config["variants"].items():
        record = read(f"numeric_{name}_run0.json")
        checks[f"process_{name}"] = record["status"] == "pass" and record["patched_tree"] == meta["series"]["tree"] and record["precision"] == {"bf16_reduced_reduction":False,"tf32":False}
        checks[f"common_provenance_{name}"] = all(record[k] == ref_record[k] for k in ("config_sha256", "workload_sha256", "patched_tree", "packages", "source_sha256"))
        data = record["replay"]
        checks[f"actual_engine_{name}"] = data["engine"]["actual_backend"] == "FlashInferBackend" and data["engine"]["dtype"] == "torch.bfloat16" and data["capture"]["buckets"] == variant["graph_bs"]
        coverage = {(r["case_id"],r["phase"]) for r in data["cases"]}
        checks[f"phase_coverage_{name}"] = coverage == {(c["case_id"],p) for c in workload["cases"] for p in ("cold","warm")} and len(data["cases"]) == 8
        request_ids = set()
        for wave in data["cases"]:
            tensors = torch.load(ROOT / wave["tensor_file"], mmap=True, weights_only=True)
            checks[f"tensor_hash_{name}_{wave['case_id']}_{wave['phase']}"] = digest(ROOT / wave["tensor_file"]) == wave["tensor_sha256"]
            for rid, target in tensors.items():
                request_ids.add(rid)
                expected = reference[rid]
                if target.shape != expected.shape:
                    raise ValueError("Missing steps or vocabulary entries")
                for step in range(config["output_tokens"]):
                    actual, ref = target[step].float(), expected[step].float()
                    finite = bool(torch.isfinite(actual).all() and torch.isfinite(ref).all())
                    close = bool(torch.isclose(actual, ref, atol=policy["atol"], rtol=policy["rtol"]).all())
                    relative = float(torch.linalg.vector_norm(actual - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-12))
                    stored = data["errors"][rid][step]
                    valid = finite and close and relative <= policy["max_relative_rmse"]
                    checks[f"metric_consistency_{name}"] = checks.get(f"metric_consistency_{name}",True) and valid == stored["passed"] and math.isclose(relative,stored["relative_rmse"],abs_tol=1e-8,rel_tol=1e-5)
                    vectors += 1
                    if not valid:
                        rows.append(dict(variant=name, request_id=rid,step=step,finite=finite,close=close,relative_rmse=relative))
            state = wave["released"]
            checks[f"released_{name}"] = checks.get(f"released_{name}",True) and not state["pending"] and not state["running"] and not state["retired"] and state["prefix_protected"] == 0 and state["tables"] == 4 and state["free_pages"] + state["prefix_total"] == state["num_pages"]
        checks[f"request_coverage_{name}"] = request_ids == expected_ids
    checks["all_7168_vectors_compared"] = vectors == len(config["variants"]) * len(expected_ids) * config["output_tokens"]
    checks["all_vectors_within_frozen_tolerance"] = not rows
    passed = all(checks.values())
    return dict(stage="I2",passed=passed,performance_allowed=passed,checks=checks,
        compared_vectors=vectors,failures=rows,config_sha256=ref_record["config_sha256"],
        workload_sha256=ref_record["workload_sha256"],patched_tree=meta["series"]["tree"],
        tolerances=policy,reference_sha256=ref_record["reference"]["tensor_sha256"],
        source_sha256=digest(__file__), evidence_sha256={p.name:digest(p) for p in directory.glob("numeric_*.json")})


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    result=audit_numerics(args.directory)
    path=args.directory/"numerical_gate.json"
    if args.check_only:
        if json.loads(path.read_text())!=result:raise ValueError("Numerical audit changed")
    elif path.exists():raise ValueError("Preserve existing numerical gate")
    else:path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("passed","compared_vectors")}))
    raise SystemExit(0 if result["passed"] else 1)

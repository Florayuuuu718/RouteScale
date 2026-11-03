"""Reopen full-training F tensors, gradients, AdamW states and clipping norm."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_f_ep import sha


def audit(directory):
    job = json.loads((directory / "job.json").read_text())
    failures, checks, errors, rows = [], 0, 0.0, []
    tolerance = {"atol": 0.002, "rtol": 0.03} if job["dtype"] == "bfloat16" else {"atol": 3e-5, "rtol": 3e-4}

    def check(ok, name):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(name)

    def compare(actual, reference, name):
        nonlocal errors
        if isinstance(reference, dict):
            check(isinstance(actual, dict) and set(actual) == set(reference), name + "/keys")
            for key in reference.keys() & actual.keys():
                compare(actual[key], reference[key], name + "/" + key)
        elif reference is None:
            check(actual is None, name + "/None")
        elif torch.is_tensor(reference):
            if not torch.is_tensor(actual) or actual.shape != reference.shape:
                check(False, name + "/shape")
                return
            check(bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()), name + "/finite")
            check(torch.allclose(actual.double(), reference.double(), **tolerance), name + "/numeric")
            if actual.numel():
                errors = max(errors, float((actual.double() - reference.double()).abs().max()))
        else:
            check(actual == reference, name + "/scalar")

    check(job["returncode"] == 0 and not job["timed_out"], "bounded successful exit")
    for path, digest in job["source_sha256"].items():
        check(sha(ROOT / path) == digest, "source/" + path)
    reports, hashes, reports_hashes = [], {}, {}
    for rank in range(job["world"]):
        path = directory / f"rank{rank}.json"
        reports_hashes[path.name] = sha(path)
        report = json.loads(path.read_text())
        reports.append(report)
        check(report["rank"] == rank and report["world"] == job["world"], f"rank{rank}/identity")
        check(report["global_tokens_per_update"] == (64 if job["tiny"] else 16384), f"rank{rank}/global tokens")
        check(report["accumulation"] == (2 if job["tiny"] else 4), f"rank{rank}/accumulation")
        check({e["mode"] for e in report["entries"]} == {"reference", "ep", "replicated"}, f"rank{rank}/modes")
        entries = {e["mode"]: e for e in report["entries"]}
        for entry in entries.values():
            hashes[entry["raw"]] = entry["sha256"]
            check(sha(entry["raw"]) == entry["sha256"], f"rank{rank}/{entry['mode']}/raw hash")
        reference = torch.load(entries["reference"]["raw"], map_location="cpu", weights_only=True)
        norm_after = sum(float(g.double().square().sum()) for g in reference["gradients"].values() if g is not None) ** 0.5
        target = min(float(reference["norm"]), 0.01 if job["tiny"] else 1.0)
        check(abs(norm_after - target) <= 2e-5 * max(1.0, target), f"rank{rank}/independent clipped norm")
        if job["tiny"]:
            check(float(reference["norm"]) > 0.01, f"rank{rank}/clipping actually active")
        prefix = f"transformer.h.{1 if job['tiny'] else 4}.mlp.experts."
        owners = {str(i) for i in range(rank * (4 // job["world"]), (rank + 1) * (4 // job["world"]))}
        for mode in ("replicated", "ep"):
            actual = torch.load(entries[mode]["raw"], map_location="cpu", weights_only=True)
            expected = dict(reference)
            names = {n for n in reference["weights"] if mode != "ep" or not n.startswith(prefix) or n[len(prefix):].split(".")[0] in owners}
            for key in ("weights", "gradients", "optimizer"):
                expected[key] = {name: value for name, value in reference[key].items() if name in names}
            compare(actual, expected, f"rank{rank}/{mode}")
            rows.append({"rank": rank, "mode": mode, "loss": float(actual["loss"]),
                         "global_grad_norm_before": float(actual["norm"]), "owned_parameters": len(names)})
            del actual, expected
            gc.collect()
        del reference
        gc.collect()
    check(len({r["window_sha256"] for r in reports}) == 1, "identical global windows across ranks")
    return {"passed": not failures, "checks": checks, "failures": failures,
            "raw_sha256": hashes, "report_sha256": reports_hashes, "job_sha256": sha(directory / "job.json"),
            "source_sha256": job["source_sha256"], "max_absolute_error": errors, "rows": rows,
            "real_multigpu": job["device"] == "cuda" and job["world"] > 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    with (args.directory / "audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ("passed", "checks", "failures", "max_absolute_error")}))
    raise SystemExit(0 if result["passed"] else 1)

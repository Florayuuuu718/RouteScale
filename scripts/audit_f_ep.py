"""Independent F audit: reopen raw tensors, reference outputs and rank ledgers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(directory, config):
    directory = Path(directory)
    manifest = json.loads((directory / "job.json").read_text())
    tolerance = config["tolerances"][manifest["dtype"]]
    checks, failures, max_error = 0, [], 0.0

    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(label)

    def compare(actual, reference, label):
        nonlocal max_error
        if isinstance(reference, dict):
            check(isinstance(actual, dict) and set(actual) == set(reference), label + "/keys")
            if not isinstance(actual, dict):
                return
            for key in reference.keys() & actual.keys():
                compare(actual[key], reference[key], f"{label}/{key}")
        elif reference is None:
            check(actual is None, label + "/absent")
        elif torch.is_tensor(reference):
            if not torch.is_tensor(actual) or actual.shape != reference.shape:
                check(False, label + "/shape")
                return
            check(bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()), label + "/finite")
            if reference.dtype in (torch.bool, torch.int64):
                check(torch.equal(actual, reference), label + "/exact")
            else:
                check(torch.allclose(actual.to(torch.float64), reference.to(torch.float64), **tolerance), label + "/numeric")
                if actual.numel():
                    max_error = max(max_error, float((actual.double() - reference.double()).abs().max()))
        else:
            check(actual == reference, label + "/scalar")

    check(manifest["returncode"] == 0 and not manifest["timed_out"], "successful bounded exit")
    check(manifest["config_sha256"] == sha(Path(__file__).resolve().parents[1] / "config/f_ep_matrix.json"), "configuration hash")
    for name, digest in manifest["source_sha256"].items():
        check(sha(Path(__file__).resolve().parents[1] / name) == digest, "current source/" + name)
    world = manifest["world"]
    reports = [json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(world)]
    report_hashes = {f"rank{rank}.json": sha(directory / f"rank{rank}.json") for rank in range(world)}
    ledgers, cases, steps, raw_hashes = {}, set(), 0, {}
    for rank, report in enumerate(reports):
        check(report["rank"] == rank and report["world"] == world, f"rank{rank}/identity")
        check(report["backend"] == ("nccl" if manifest["device"] == "cuda" else "gloo"), f"rank{rank}/backend")
        expected_cases = {"gpt"} if manifest["gpt"] else ({"balanced"} if manifest["formal"] or manifest["trace"] else set(config["cases"]))
        check({entry["case"] for entry in report["entries"]} == expected_cases, f"rank{rank}/all cases")
        check(len(report["entries"]) == len(expected_cases), f"rank{rank}/unique cases")
        for entry in report["entries"]:
            path = Path(entry["raw"])
            check(sha(path) == entry["sha256"], f"rank{rank}/{entry['case']}/raw hash")
            raw_hashes[str(path)] = entry["sha256"]
            raw = torch.load(path, map_location="cpu", weights_only=True)
            cases.add(entry["case"])
            if entry["case"] == "gpt":
                prefix = "transformer.h.1.mlp.experts."
                owners = {str(i) for i in range(rank * (4 // world), (rank + 1) * (4 // world))}
                expected_names = {name for name in raw["reference"]["weights"]
                                  if not name.startswith(prefix) or name[len(prefix):].split(".")[0] in owners}
                check(set(raw["actual"]["weights"]) == expected_names, f"GPT/rank{rank}/owned names")
                ref = dict(raw["reference"])
                for key in ("gradients", "weights", "optimizer"):
                    ref[key] = {k: v for k, v in ref[key].items() if k in expected_names}
                compare(raw["actual"], ref, f"GPT/rank{rank}")
                steps += 1
                continue
            expected_owners = list(map(str, range(rank * (4 // world), (rank + 1) * (4 // world))))
            check(entry["owned_experts"] == expected_owners, f"rank{rank}/expert ownership")
            check(len(raw) == entry["steps"], f"rank{rank}/update count")
            check(len(raw) == (1 if manifest["formal"] else config["updates"]), f"rank{rank}/frozen updates")
            previous = None
            for record in raw:
                steps += 1
                label = f"rank{rank}/{entry['case']}/{record['step']}"
                actual, ref, rep = (record[k] for k in ("actual", "reference", "replicated"))
                expected_names = {name for name in ref["weights"] if name.startswith("router.") or name.split(".")[1] in expected_owners}
                check(set(actual["weights"]) == expected_names, label + "/owned parameter names")
                for key in ("output", "input_grad", "counts", "processed", "indices", "drops"):
                    compare(actual[key], ref[key], label + "/" + key)
                for key in ("gradients", "weights", "optimizer"):
                    compare(actual[key], {k: v for k, v in ref[key].items() if k in expected_names}, label + "/EP/" + key)
                for key in rep:
                    compare(rep[key], ref[key], label + "/replicated/" + key)
                counts = ref["counts"]
                check(int(counts.sum()) == record["total"] == sum(record["sizes"]), label + "/token conservation")
                cap = record["capacity"]
                if cap is not None:
                    factor = config["formal_shape"]["capacity_factor"] if manifest["formal"] else 0.5
                    check(cap == math.ceil(factor * record["total"] / 4), label + "/global capacity")
                accepted = ~ref["drops"]
                expected_send = torch.bincount(ref["indices"][accepted] // (4 // world), minlength=world).tolist()
                check(record["send"] == expected_send, label + "/send counts")
                check(sum(record["send"]) == int(accepted.sum()), label + "/accepted tokens")
                for name in expected_names:
                    if name.startswith("experts.") and ref["gradients"][name] is None:
                        check(actual["gradients"][name] is None, label + "/unused gradient absent/" + name)
                        if previous is not None:
                            check(torch.equal(actual["weights"][name], previous["weights"][name]), label + "/unused weights unchanged/" + name)
                            compare(actual["optimizer"][name], previous["optimizer"][name], label + "/unused optimizer unchanged/" + name)
                ledgers[(entry["case"], record["step"], rank)] = {
                    "send": record["send"], "receive": record["receive"], "width": record["width"],
                    "processed": int(ref["processed"].sum()), "tokens": record["total"]}
                previous = actual
    for (case, step, rank), ledger in ledgers.items():
        for peer in range(world):
            check(ledger["send"][peer] == ledgers[(case, step, peer)]["receive"][rank], f"{case}/{step}/split transpose/{rank}/{peer}")
    traffic = []
    for case, step in sorted({(c, s) for c, s, _ in ledgers}):
        rows = [ledgers[(case, step, rank)] for rank in range(world)]
        remote = sum(row["send"][peer] for rank, row in enumerate(rows) for peer in range(world) if peer != rank)
        # Four floating activation exchanges (two forward, two backward), plus
        # one int64 expert ID. Excludes count/split/router-grad collectives.
        size = 2 if manifest["dtype"] == "bfloat16" else (8 if manifest["dtype"] == "float64" else 4)
        input_size = 8 if manifest["dtype"] == "float64" else 4
        payload = remote * (2 * rows[0]["width"] * (size + input_size) + 8)
        traffic.append({"case": case, "step": step, "remote_tokens": remote,
                        "activation_and_id_logical_bytes": payload, "wire_bytes_measured": False,
                        "send_matrix": [row["send"] for row in rows]})
    expected_cases = {"gpt"} if manifest["gpt"] else ({"balanced"} if manifest["formal"] or manifest["trace"] else set(config["cases"]))
    check(cases == expected_cases, "complete case coverage")
    result = {"passed": not failures, "checks": checks, "failures": failures,
              "cases": sorted(cases), "rank_updates": steps, "max_absolute_error": max_error,
              "tolerance": tolerance, "traffic": traffic, "raw_sha256": raw_hashes,
              "job_sha256": sha(directory / "job.json"), "report_sha256": report_hashes,
              "world": world, "device": manifest["device"],
              "cross_gpu_accepted": manifest["device"] == "cuda" and world > 1 and not failures}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((Path(__file__).resolve().parents[1] / "config/f_ep_matrix.json").read_text())
    result = audit(args.directory, config)
    output = args.output or args.directory / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "checks", "failures", "rank_updates", "max_absolute_error")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

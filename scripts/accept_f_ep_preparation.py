"""Archive and accept only F's bounded local preparation, never full F/EP2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare_f_ep import CONFIG, ROOT, SOURCES, jobs, require_idle_gpus


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(runtime, historical):
    checks, failures, audits = 0, [], []

    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(message)

    for job in jobs():
        directory = runtime / job["label"]
        manifest = json.loads((directory / "job.json").read_text())
        audit = json.loads((directory / "audit.json").read_text())
        check(all(manifest[k] == v for k, v in job.items()), job["label"] + "/scope")
        check(manifest["returncode"] == 0 and not manifest["timed_out"], job["label"] + "/exit")
        check(audit["passed"] and not audit["failures"], job["label"] + "/audit")
        check(sha(directory / "job.json") == audit["job_sha256"], job["label"] + "/job hash")
        check(manifest["config_sha256"] == sha(CONFIG), job["label"] + "/config")
        check(set(manifest["source_sha256"]) == set(SOURCES), job["label"] + "/source set")
        for path, digest in manifest["source_sha256"].items():
            check(sha(ROOT / path) == digest, job["label"] + "/source/" + path)
        for path, digest in audit["raw_sha256"].items():
            check(sha(path) == digest, job["label"] + "/raw/" + Path(path).name)
        for path, digest in audit["report_sha256"].items():
            check(sha(directory / path) == digest, job["label"] + "/report/" + path)
        if job["device"] == "cuda":
            check(all(not memory and not utilization for _, memory, utilization in manifest["gpu_after"]), job["label"] + "/GPU released")
        audits.append({"label": job["label"], **{k: audit[k] for k in ("checks", "rank_updates", "max_absolute_error", "cases", "world", "device")}})
    history = json.loads(historical.read_text())
    immutable = {name: digest for name, digest in history.items() if name != "results/README.md"}
    for name, digest in immutable.items():
        check(sha(ROOT / name) == digest, "historical evidence changed/" + name)
    cpu = json.loads((runtime / "cpu_checks.json").read_text())
    check(cpu["passed"] and cpu["tests"] >= 60, "CPU contracts/regressions")
    require_idle_gpus(1)
    check(True, "GPU currently idle")
    result = {"scope": "F_local_preparation_v1", "passed": not failures,
              "local_preparation_complete": not failures, "f_stage_complete": False,
              "real_multigpu_accepted": False, "paid_resources_used": False,
              "performance_execution_allowed": False, "checks": checks, "failures": failures,
              "numerical_audit_checks": sum(a["checks"] for a in audits),
              "rank_updates": sum(a["rank_updates"] for a in audits), "jobs": audits,
              "historical_immutable_files": len(immutable), "historical_sha256": sha(historical),
              "cpu_checks": cpu, "pending_host_qualifications": jobs(False),
              "source_sha256": {name: sha(ROOT / name) for name in SOURCES}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = inspect(args.runtime, args.historical)
    if not result["passed"]:
        print(json.dumps(result))
        raise SystemExit(1)
    if args.check_only:
        original = json.loads((args.output / "acceptance.json").read_text())
        for name, digest in original["evidence_sha256"].items():
            if sha(args.output / name) != digest:
                raise ValueError("changed archived evidence: " + name)
        if any(original[k] != result[k] for k in result):
            raise ValueError("acceptance has changed")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for job in jobs():
            source = args.runtime / job["label"]
            target = args.output / job["label"]
            target.mkdir()
            for name in ("job.json", "audit.json", "worker.log", "audit.log", *[f"rank{r}.json" for r in range(job["world"])]):
                shutil.copyfile(source / name, target / name)
        for name in SOURCES:
            path = args.output / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        shutil.copyfile(args.historical, args.output / "historical_before.json")
        shutil.copyfile(args.runtime / "cpu_checks.json", args.output / "cpu_checks.json")
        result["evidence_sha256"] = {str(path.relative_to(args.output)): sha(path)
                                     for path in args.output.rglob("*") if path.is_file()}
        (args.output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "checks", "numerical_audit_checks", "rank_updates", "f_stage_complete")}))


if __name__ == "__main__":
    main()

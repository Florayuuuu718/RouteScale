"""Accept and archive the additional F complete-training local qualifiers."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_f_training import SOURCES, jobs
from prepare_f_ep import require_idle_gpus, sha


def inspect(runtime):
    entries, checks, failures = [], 0, []
    def check(ok, name):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(name)
    for job in jobs():
        path = runtime / job["label"]
        manifest = json.loads((path / "job.json").read_text())
        result = json.loads((path / "audit.json").read_text())
        check(all(manifest[k] == v for k, v in job.items()), job["label"] + "/scope")
        check(manifest["returncode"] == 0 and not manifest["timed_out"], job["label"] + "/exit")
        check(result["passed"] and not result["failures"], job["label"] + "/audit")
        check(sha(path / "job.json") == result["job_sha256"], job["label"] + "/job hash")
        check(set(manifest["source_sha256"]) == set(SOURCES), job["label"] + "/source set")
        for name, digest in manifest["source_sha256"].items():
            check(sha(ROOT / name) == digest, job["label"] + "/source/" + name)
        for name, digest in result["raw_sha256"].items():
            check(sha(name) == digest, job["label"] + "/raw/" + Path(name).name)
        for name, digest in result["report_sha256"].items():
            check(sha(path / name) == digest, job["label"] + "/report/" + name)
        if job["device"] == "cuda":
            check(all(not memory and not utilization for _, memory, utilization in manifest["gpu_after"]), job["label"] + "/GPU released")
        entries.append({"label": job["label"], "checks": result["checks"], "max_absolute_error": result["max_absolute_error"], "rows": result["rows"]})
    require_idle_gpus(1)
    check(True, "GPU currently idle")
    return dict(scope="F_complete_training_local_v1", passed=not failures,
                checks=checks, failures=failures, jobs=entries,
                numerical_checks=sum(entry["checks"] for entry in entries),
                paid_resources_used=False, full_multigpu_accepted=False,
                performance_execution_allowed=False, performance_processes=0,
                source_sha256={name: sha(ROOT/name) for name in SOURCES})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = inspect(args.runtime)
    if not result["passed"]:
        print(json.dumps(result)); raise SystemExit(1)
    if args.check_only:
        original = json.loads((args.output/"acceptance.json").read_text())
        for name,digest in original["evidence_sha256"].items():
            if sha(args.output/name) != digest:raise ValueError("changed archived training evidence")
        if any(original[k] != result[k] for k in result):raise ValueError("training acceptance changed")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for job in jobs():
            target=args.output/job["label"];target.mkdir()
            for name in ("job.json","audit.json","worker.log","audit.log",*[f"rank{r}.json" for r in range(job["world"])]):
                shutil.copyfile(args.runtime/job["label"]/name,target/name)
        for name in (*SOURCES,"scripts/collect_f_ep_trace.py","scripts/accept_f_training_preparation.py"):
            target=args.output/"source_snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        result["evidence_sha256"]={str(p.relative_to(args.output)):sha(p) for p in args.output.rglob("*") if p.is_file()}
        (args.output/"acceptance.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("passed","checks","numerical_checks","full_multigpu_accepted")}))

"""Export compact trace/negative/preflight evidence without moving raw tensors."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
runtime = ROOT / "runs/f_ep/local_v1"
output = ROOT / "results/f_ep/local_v1"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

inventory = []
for label in ("cpu2_trace", "cuda1_trace"):
    reports = sorted((runtime / label).glob("rank*.json"))
    for path in reports:
        report = json.loads(path.read_text())
        trace = Path(report["trace"])
        content = json.loads(trace.read_text())
        categories = Counter(e.get("cat", "metadata") for e in content["traceEvents"])
        assert any(e.get("name", "").startswith("ep/") for e in content["traceEvents"])
        if report["device"].startswith("cuda"):
            assert categories["kernel"] > 0
        inventory.append({"job": label, "rank": report["rank"], "backend": report["backend"],
                          "trace_sha256": sha(trace), "trace_bytes": trace.stat().st_size,
                          "categories": dict(categories), "cross_gpu_communication": False,
                          "timing_is_profiled_and_contains_cold_start": True})
(output / "trace_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
for source, target in ((ROOT/"runs/f_ep/negative_controls001/controls.json", "negative_controls.json"),
                       (ROOT/"runs/f_ep/check_audit_controls.py", "check_audit_controls.py"),
                       (Path(__file__), "export_preparation.py"),
                       (ROOT/"runs/f_ep/dataset_preflight.json", "dataset_preflight.json"),
                       (ROOT/"config/f_ep_rental_requirements.json", "rental_requirements.json"),
                       (ROOT/"runs/f_ep/local_v1_plan.json", "local_plan.json"),
                       (ROOT/"runs/f_ep/delivery_cpu_qualified.log", "delivery_cpu.log"),
                       (ROOT/"runs/f_ep/moe_regression.log", "moe_regression.log")):
    assert not (output/target).exists()
    shutil.copyfile(source, output/target)
issues = output / "preparation_issues"
issues.mkdir()
shutil.copyfile(ROOT/"runs/f_ep/debug001/cpu2/worker.log", issues/"resolved_interpreter_lost_venv.log")
shutil.copyfile(ROOT/"runs/f_ep/debug001/cpu2/job.json", issues/"resolved_interpreter_lost_venv.json")
shutil.copyfile(ROOT/"runs/f_ep/delivery_cpu.log", issues/"system_python_missing_httpx.log")
print(json.dumps({"traces": len(inventory), "passed": True}))

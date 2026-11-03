"""Reproducible negative controls; intentionally corrupt disposable tensor copies."""
import copy
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_f_ep import audit, sha

runtime = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=False)
config = json.loads((ROOT / "config/f_ep_matrix.json").read_text())
results = []
for label, case in (("wrong_output", "balanced"), ("wrong_split", "balanced"), ("absent_gradient_replaced_by_zero", "all_to_one")):
    target = output / label
    target.mkdir()
    for name in ("job.json", "rank0.json"):
        shutil.copyfile(runtime / name, target / name)
    report = json.loads((target / "rank0.json").read_text())
    entry = next(e for e in report["entries"] if e["case"] == case)
    records = torch.load(entry["raw"], map_location="cpu", weights_only=True)
    if label == "wrong_output":
        records[0]["actual"]["output"][0, 0] += 1
    elif label == "wrong_split":
        records[0]["send"][0] += 1
    else:
        name = "experts.1.c_fc.weight"
        assert records[0]["actual"]["gradients"][name] is None
        records[0]["actual"]["gradients"][name] = torch.zeros_like(records[0]["actual"]["weights"][name])
    raw = target / "tampered.pt"
    torch.save(records, raw)
    # Rebind hashes deliberately, proving the audit checks tensor semantics
    # rather than rejecting these fixtures solely for a changed file hash.
    entry["raw"], entry["sha256"] = str(raw), sha(raw)
    (target / "rank0.json").write_text(json.dumps(report, indent=2) + "\n")
    result = audit(target, config)
    assert not result["passed"], label
    assert not any("hash" in f for f in result["failures"]), result["failures"]
    results.append({"control": label, "rejected": True, "checks": result["checks"], "failures": result["failures"]})
(output / "controls.json").write_text(json.dumps({"passed": True, "controls": results}, indent=2) + "\n")
print(json.dumps({"passed": True, "rejected_semantic_controls": len(results)}))

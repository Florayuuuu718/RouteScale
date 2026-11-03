"""Collect a qualified two-GPU D/E-shape EP trace, separately from timing."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_f_ep import run_job, sha
from prepare_f_training import performance_gate
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    args = parser.parse_args()
    performance_gate(args.host_root)
    config = json.loads((ROOT / "config/f_ep_matrix.json").read_text())
    result = run_job(dict(label="cuda2_formal_trace", world=2, device="cuda", dtype="bfloat16",
                          formal=True, gpt=False, trace=True, trace_launcher_sha256=sha(__file__)),
                     args.output, args.python, config)
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}))

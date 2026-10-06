#!/usr/bin/env python3
"""Verify that a split native C5 run exactly matches an uninterrupted run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from distributed_common import atomic_json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", default="results/c5_coverage/resume_check.json")
    return parser.parse_args()


def run(arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "train_coverage.py"), *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    dtype = "float32"
    with tempfile.TemporaryDirectory(prefix="routescale-c5-resume-") as temporary:
        root = Path(temporary)
        split_dir = root / "split"
        reference_dir = root / "reference"
        common = [
            f"--device={args.device}",
            f"--dtype={dtype}",
            "--data-dir=data/tinystories_debug",
            "--block-size=32",
            "--batch-size=2",
            "--global-windows-per-update=4",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64",
            "--eval-windows=4",
        ]
        run([*common, "--max-updates=2", "--checkpoint-every=1",
             f"--out-dir={split_dir}"])
        run([*common, "--max-updates=4", "--checkpoint-every=1",
             f"--resume-from={split_dir / 'checkpoint.pt'}",
             f"--out-dir={split_dir}"])
        run([*common, "--max-updates=4", "--checkpoint-every=0",
             f"--out-dir={reference_dir}"])

        split = torch.load(
            split_dir / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        reference = torch.load(
            reference_dir / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        checks = {
            "model_state_sha256_equal": (
                split["model_state_sha256"] == reference["model_state_sha256"]
            ),
            "loss_history_equal": split["loss_history"] == reference["loss_history"],
            "next_update_equal": split["next_update"] == reference["next_update"] == 4,
            "window_prefix_sha256_equal": (
                split["global_window_prefix_sha256"]
                == reference["global_window_prefix_sha256"]
            ),
        }
        result = {
            "schema_version": 1,
            "stage": "C5",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if all(checks.values()) else "fail",
            "device": args.device,
            "dtype": dtype,
            "split": "2 updates + resume to 4 updates",
            "reference": "4 uninterrupted updates",
            "checks": checks,
            "loss_history": split["loss_history"],
            "model_state_sha256": split["model_state_sha256"],
            "global_window_prefix_sha256": split["global_window_prefix_sha256"],
        }
        atomic_json_dump(result, args.output)
        print(json.dumps(result, indent=2))
        if result["status"] != "pass":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

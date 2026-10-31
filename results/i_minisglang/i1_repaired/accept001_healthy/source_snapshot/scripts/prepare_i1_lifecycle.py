"""Replay the I1 patch series without modifying the pinned upstream checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "patches/minisgl-i1/series.json"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def validate(checkout):
    series = json.loads(SERIES.read_text())
    for name, digest in series["patch_sha256"].items():
        if hashlib.sha256((SERIES.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Patch hash mismatch: {name}")
    if git(checkout, "status", "--porcelain"):
        raise ValueError("Repaired checkout must be clean")
    if git(checkout, "rev-parse", "HEAD^{tree}") != series["tree"]:
        raise ValueError("Repaired checkout tree differs from frozen patch series")
    if git(checkout, "rev-parse", "HEAD~7") != series["base_commit"]:
        raise ValueError("Expected exactly seven commits on the pinned upstream")
    return series


def prepare(checkout):
    series = json.loads(SERIES.read_text())
    if not checkout.exists():
        upstream = ROOT / "runs/vendor/mini-sglang"
        if git(upstream, "status", "--porcelain"):
            raise ValueError("Preserve a clean upstream checkout")
        git(upstream, "worktree", "add", "--detach", str(checkout), series["base_commit"])
        git(checkout, "-c", "user.name=RouteScale", "-c", "user.email=routescale@localhost",
            "-c", "commit.gpgSign=false", "am",
            *(str(SERIES.parent / name) for name in series["patch_sha256"]))
    return validate(checkout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=ROOT / "runs/vendor/mini-sglang-i1")
    args = parser.parse_args()
    print(json.dumps(dict(checkout=str(args.checkout), **prepare(args.checkout))))

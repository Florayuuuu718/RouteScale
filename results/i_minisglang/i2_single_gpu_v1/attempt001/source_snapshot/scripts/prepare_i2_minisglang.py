"""Replay I2 extensions on the preserved I1 source tree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_i1_lifecycle import prepare, git
from serving.i2_workloads import digest


def validate(checkout):
    series = json.loads((ROOT / "patches/minisgl-i2/series.json").read_text())
    if git(checkout, "status", "--porcelain") or git(checkout, "rev-parse", "HEAD^{tree}") != series["tree"]:
        raise ValueError("Expected clean, frozen I2 checkout")
    if git(checkout, "rev-parse", "HEAD~2^{tree}") != series["base_i1_tree"]:
        raise ValueError("I2 patches must retain the accepted I1 base")
    if any(digest(ROOT / "patches/minisgl-i2" / name) != value for name, value in series["patch_sha256"].items()):
        raise ValueError("I2 patch hash mismatch")
    return series


def prepare_i2():
    i1 = ROOT / "runs/vendor/mini-sglang-i1-fixed"
    prepare(i1)
    checkout = ROOT / "runs/vendor/mini-sglang-i2"
    if not checkout.exists():
        git(i1, "worktree", "add", "--detach", str(checkout), "HEAD")
        patches = sorted((ROOT / "patches/minisgl-i2").glob("*.patch"))
        git(checkout, "-c", "user.name=RouteScale", "-c", "user.email=routescale@localhost", "am", *(str(p) for p in patches))
    return validate(checkout)


if __name__ == "__main__":
    print(json.dumps(prepare_i2()))

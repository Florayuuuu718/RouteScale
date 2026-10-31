"""Make a local CUDA 13.0 header compatibility copy; never edit system CUDA."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/usr/local/cuda-13.0")
TARGET = ROOT / "runs/toolchains/cuda-13.0-i1"
RELATIVE = Path("targets/x86_64-linux/include/crt/math_functions.h")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_hash = digest(SOURCE / RELATIVE)
    if (TARGET / "routescale_patch.json").exists():
        record = json.loads((TARGET / "routescale_patch.json").read_text())
        if digest(TARGET / RELATIVE) != record["patched_sha256"] or original_hash != record["system_sha256"]:
            raise ValueError("Toolchain has changed")
        return
    original = (SOURCE / RELATIVE).read_text()
    patched = original
    for declaration in ("rsqrt(double x);", "rsqrtf(float x);"):
        if patched.count(declaration) != 1:
            raise ValueError(f"Expected exactly one declaration: {declaration}")
        patched = patched.replace(declaration, declaration[:-1] + " noexcept (true);")
    if TARGET.exists():
        # Only our incomplete preparation tree; never follow/remove source symlinks.
        shutil.rmtree(TARGET)
    # Copy headers and nvcc, link the remaining read-only SDK files to avoid a 5 GiB copy.
    TARGET.mkdir(parents=True)
    for item in SOURCE.iterdir():
        if item.name not in {"bin", "targets", "include"}:
            (TARGET / item.name).symlink_to(item, target_is_directory=item.is_dir())
    (TARGET / "bin").mkdir()
    for item in (SOURCE / "bin").iterdir():
        if item.name == "nvcc":
            shutil.copy2(item, TARGET / "bin/nvcc")
        else:
            (TARGET / "bin" / item.name).symlink_to(item)
    for target in (SOURCE / "targets").iterdir():
        destination = TARGET / "targets" / target.name
        destination.mkdir(parents=True)
        for item in target.iterdir():
            if item.name == "include":
                shutil.copytree(item, destination / "include", symlinks=True)
            else:
                (destination / item.name).symlink_to(item, target_is_directory=item.is_dir())
    (TARGET / "include").symlink_to("targets/x86_64-linux/include")
    header = TARGET / RELATIVE
    header.write_text(patched)
    if digest(SOURCE / RELATIVE) != original_hash:
        raise ValueError("System CUDA must remain untouched")
    record = dict(system_path=str(SOURCE), system_sha256=original_hash,
        patched_sha256=digest(header), patched_header=str(RELATIVE),
        changes=["rsqrt(double x) noexcept (true)", "rsqrtf(float x) noexcept (true)"],
        reason="CUDA 13.0 declarations conflict with local glibc 2.43 exception specifications",
        reference="https://forums.developer.nvidia.com/t/cuda-headers-in-crt-math-functions-h-still-broken-in-debian-13-repo/362940")
    (TARGET / "routescale_patch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

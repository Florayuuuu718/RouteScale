#!/usr/bin/env python3
"""Package small C-stage evidence while excluding data and bulky artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile


ALLOWED_SUFFIXES = {".json", ".log", ".txt", ".md", ".csv"}
EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    "checkpoints",
    "runtime",
    "traces",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-file-mib", type=float, default=16.0)
    return parser.parse_args()


def include_file(path: Path, root: Path, max_bytes: int) -> tuple[bool, str]:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts[:-1]):
        return False, "excluded_directory"
    if path.name.endswith("_trace.json") or path.name.endswith("_trace.json.gz"):
        return False, "profiler_trace"
    if path.suffix.casefold() not in ALLOWED_SUFFIXES:
        return False, "unsupported_suffix"
    if path.stat().st_size > max_bytes:
        return False, "over_size_limit"
    return True, "included"


def main() -> None:
    args = parse_args()
    root = args.results_dir.resolve()
    max_bytes = int(args.max_file_mib * 1024**2)
    included = []
    skipped: dict[str, int] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        keep, reason = include_file(path, root, max_bytes)
        if not keep:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        data = path.read_bytes()
        included.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(root),
        "max_file_mib": args.max_file_mib,
        "included_file_count": len(included),
        "included_total_bytes": sum(item["bytes"] for item in included),
        "skipped_file_counts": skipped,
        "files": included,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for item in included:
            archive.add(root / item["path"], arcname=f"results/{item['path']}")
        encoded = (
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        info = tarfile.TarInfo("artifact_manifest.json")
        info.size = len(encoded)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(encoded))
    print(
        f"packaged {len(included)} files, "
        f"{manifest['included_total_bytes'] / 1024**2:.2f} MiB before compression"
    )
    print(f"skipped {skipped}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

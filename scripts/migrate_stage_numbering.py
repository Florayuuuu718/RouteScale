"""Migrate serving stages once and verify the resulting paths and evidence.

H/I/J/K become G/H/I/J. The manifest records byte hashes before and after
renaming; a local archive preserves original text. No experiment is rerun.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("results/stage_numbering_migration/manifest.json")
ARCHIVE = Path("runs/archives/stage_numbering_before_20261024.tar.gz")
LETTER_MAP = dict(zip("hijkHIJK", "ghijGHIJ"))
LOWER = re.compile(
    r"h_serving\b|i_nanovllm\b|j_minisglang\b|(?<![A-Za-z0-9])[h-k][123](?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[hi]_stage(?![A-Za-z0-9])"
)
TASK = re.compile(r"(?<![A-Za-z0-9])[H-K][123](?![A-Za-z0-9])")
STAGE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[H-K](?:[/–—-][H-K])+|[H-K](?= 阶段|[- ]stage\b))"
)
SHA = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
JSON_STRING = re.compile(rb'"(?:\\.|[^"\\])*"')
CURRENT_DOCS = {
    "README.md", "results/README.md", "docs/RouteScale_实施教程大纲.md",
    "docs/RouteScale_AIInfra_项目规划书.md",
}
EXCLUDED = {
    str(MANIFEST.parent), "scripts/migrate_stage_numbering.py",
    "tests/test_stage_numbering_migration.py", "docs/stage_numbering_migration.md",
}
LOCAL_DIRS = {"checkpoints", "workloads", "validation", "logs", "cache", "tmp", "downloads", "traces"}
TEXT_EXTENSIONS = {".py", ".json", ".md", ".txt", ".log", ".lock", ".sh"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rename_identifiers(text: str) -> str:
    return LOWER.sub(lambda m: LETTER_MAP[m[0][0]] + m[0][1:], text)


def translate_text(text: str, path: str) -> str:
    text = rename_identifiers(text)
    if path in CURRENT_DOCS:
        return text
    # H1/H2 in the E review are hypothesis names; H2D and H100 are hardware terms.
    stage_scope = (
        path.startswith(("serving/", "scripts/", "tests/", "config/serving/",
                         "results/h_serving/", "results/i_nanovllm/", "runs/"))
        or (path.startswith("docs/") and (LOWER.search(path) or "nanovllm" in path))
    )
    if stage_scope:
        text = TASK.sub(lambda m: LETTER_MAP[m[0][0]] + m[0][1:], text)
        text = STAGE.sub(lambda m: "".join(LETTER_MAP.get(c, c) for c in m[0]), text)
    return text


def rewrite_hashes(data: bytes, mapping: dict[str, str]) -> bytes:
    return SHA.sub(lambda m: mapping.get(m[0].decode(), m[0].decode()).encode(), data)


def settle_hashes(records: list[dict]) -> dict[str, str]:
    """Rebind file checksums from original templates until the DAG is stable."""
    mapping = {}
    for _ in range(len(records) + 1):
        updated = {}
        for record in records:
            if record["kind"] != "text":
                continue
            content = rewrite_hashes(record["template"], mapping)
            new_hash = sha256(content)
            old_hash = record["original_sha256"]
            if new_hash != old_hash:
                if old_hash in updated and updated[old_hash] != new_hash:
                    raise ValueError("Identical originals produced different migrated bytes")
                updated[old_hash] = new_hash
            record["content"] = content
            record["sha256"] = new_hash
        if updated == mapping:
            return updated
        mapping = updated
    raise ValueError("Checksum dependency cycle; no files have been moved")


def candidate_paths(root: Path):
    for directory in ("serving", "scripts", "tests", "config", "docs", "results"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if (path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
                    and not any(relative == p or relative.startswith(p + "/") for p in EXCLUDED)):
                yield path
    yield root / "README.md"
    yield root / "runs/README.md"
    for directory in LOCAL_DIRS:
        for path in (root / "runs" / directory).rglob("*"):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            # The previous turn's backup is an immutable pre-migration snapshot.
            if relative.startswith("runs/tmp/stage_renumber_"):
                continue
            if rename_identifiers(relative) != relative or directory in {"logs", "workloads"}:
                yield path


def apply_migration(root: Path):
    if (root / MANIFEST).exists():
        raise ValueError("Migration already recorded; use --check-only")
    if (root / ARCHIVE).exists():
        raise ValueError("Preserve the existing original archive")
    empty_directories = []
    for directory in LOCAL_DIRS:
        for path in (root / "runs" / directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if relative.startswith("runs/tmp/stage_renumber_"):
                continue
            target = rename_identifiers(relative)
            if path.is_dir() and not path.is_symlink() and relative != target and not any(path.iterdir()):
                empty_directories.append(dict(original_path=relative, path=target))
    records = []
    for path in sorted(set(candidate_paths(root))):
        old_path = path.relative_to(root).as_posix()
        new_path = rename_identifiers(old_path)
        opaque = old_path.startswith("runs/cache/") or path.suffix not in TEXT_EXTENSIONS
        original = None if opaque else path.read_bytes()
        if original is not None:
            try:
                template = translate_text(original.decode("utf-8"), old_path).encode("utf-8")
            except UnicodeDecodeError:
                original = None
        if original is None:
            if old_path == new_path:
                continue
            record = dict(original_path=old_path, path=new_path, kind="binary",
                          original_sha256=file_sha256(path), bytes=path.stat().st_size)
            record["sha256"] = record["original_sha256"]
        else:
            record = dict(original_path=old_path, path=new_path, kind="text",
                          original_sha256=sha256(original), bytes=len(original),
                          original=original, template=template)
        records.append(record)
    mapping = settle_hashes(records)
    records = [r for r in records if r["path"] != r["original_path"]
               or r["sha256"] != r["original_sha256"]]
    destinations = {r["path"] for r in records}
    if len(destinations) != len(records):
        raise ValueError("Migration would merge paths")
    sources = {r["original_path"] for r in records}
    for record in records:
        target = root / record["path"]
        if target.exists() and record["path"] not in sources:
            raise ValueError(f"Destination already exists: {target}")
        if record["kind"] == "text" and target.suffix == ".json":
            json.loads(record["content"])
            if JSON_STRING.sub(b'""', record["original"]) != JSON_STRING.sub(b'""', record["content"]):
                raise ValueError("Migration would change non-string JSON measurements")
    archive = root / ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as saved:
        for record in records:
            if record["kind"] == "text":
                saved.add(root / record["original_path"], arcname=record["original_path"], recursive=False)
    # Stage every source first so existing H names cannot collide with new H names.
    old_parents = set()
    with tempfile.TemporaryDirectory(prefix="stage-numbering-", dir=root / "runs/tmp") as staging:
        for index, record in enumerate(records):
            source = root / record["original_path"]
            old_parents.update(p for p in source.parents if p != root and root in p.parents)
            source.replace(Path(staging) / str(index))
        for index, record in enumerate(records):
            target = root / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            source = Path(staging) / str(index)
            if record["kind"] == "text":
                source.write_bytes(record["content"])
            source.replace(target)
    for parent in sorted(old_parents, key=lambda p: len(p.parts), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            pass
    for directory in ("serving", "scripts", "tests"):
        for cache in (root / directory).rglob("__pycache__"):
            shutil.rmtree(cache)
    for directory in empty_directories:
        (root / directory["path"]).mkdir(parents=True, exist_ok=True)
        original = root / directory["original_path"]
        while original != root and rename_identifiers(original.relative_to(root).as_posix()) != original.relative_to(root).as_posix():
            try:
                original.rmdir()
            except (FileNotFoundError, OSError):
                break
            original = original.parent
    for record in records:
        for key in ("original", "template", "content"):
            record.pop(key, None)
    manifest = dict(
        schema_version=1, evidence_class="stage_numbering_migration", status="pass",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        stage_mapping={"H": "G", "I": "H", "J": "I", "K": "J"},
        original_text_archive=dict(path=str(ARCHIVE), sha256=file_sha256(archive)),
        experiments_rerun=False, numerical_measurements_changed=False,
        file_count=len(records), renamed_files=sum(r["path"] != r["original_path"] for r in records),
        json_files=sum(r["path"].endswith(".json") for r in records),
        binary_files_unchanged=sum(r["kind"] == "binary" for r in records),
        hash_rewrites=mapping, files=records, empty_directories=empty_directories,
        limitations="Metadata and file checksums were translated, not freshly measured. "
                    "Model weights, reference tensors, timings, thresholds, status and timestamps are preserved.",
    )
    destination = root / MANIFEST
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "file_count", "renamed_files", "json_files", "binary_files_unchanged")}, indent=2), flush=True)


def check_migration(root: Path):
    manifest = json.loads((root / MANIFEST).read_text())
    for directory in manifest.get("empty_directories", []):
        if not (root / directory["path"]).is_dir():
            raise ValueError(f"Migrated directory missing: {directory['path']}")
    for record in manifest["files"]:
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Migrated file changed: {record['path']}")
        if record["kind"] == "binary" and record["sha256"] != record["original_sha256"]:
            raise ValueError("Binary bytes changed")
    archive = root / manifest["original_text_archive"]["path"]
    verified_originals = 0
    if archive.exists():
        if file_sha256(archive) != manifest["original_text_archive"]["sha256"]:
            raise ValueError("Original archive changed")
        with tarfile.open(archive, "r:gz") as saved:
            for record in manifest["files"]:
                if record["kind"] != "text":
                    continue
                original = saved.extractfile(record["original_path"]).read()
                if sha256(original) != record["original_sha256"]:
                    raise ValueError("Original archived bytes changed")
                migrated = rewrite_hashes(translate_text(original.decode("utf-8"), record["original_path"]).encode("utf-8"), manifest["hash_rewrites"])
                if sha256(migrated) != record["sha256"]:
                    raise ValueError(f"Change exceeds the recorded migration: {record['path']}")
                if record["path"].endswith(".json"):
                    if JSON_STRING.sub(b'""', original) != JSON_STRING.sub(b'""', migrated):
                        raise ValueError("Numeric measurements changed")
                verified_originals += 1
    print(f"Migration integrity: pass; {manifest['file_count']} current hashes, "
          f"{verified_originals} archived originals; no numeric or binary changes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply_migration(ROOT)
    else:
        check_migration(ROOT)


if __name__ == "__main__":
    main()

"""Pin validation and provenance shared by nano-vLLM experiment entry points."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/serving/h1_nanovllm.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(path.read_text())
    repo = ROOT / config["upstream"]["path"]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if commit != config["upstream"]["commit"] or dirty:
        raise RuntimeError(f"nano-vLLM must be a clean checkout of the pinned commit: {commit}, {dirty}")
    sys.path.insert(0, str(repo))
    return config


def base_result(config: dict, evidence_class: str) -> dict:
    repo = ROOT / config["upstream"]["path"]
    return {
        "schema_version": 1,
        "stage": "H1",
        "evidence_class": evidence_class,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": config["upstream"],
        "upstream_source_sha256": {
            str(path.relative_to(repo)): digest(path)
            for path in sorted(repo.rglob("*.py")) if ".git" not in path.parts
        },
        "upstream_license_sha256": digest(repo / "LICENSE"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": json.loads(subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"], text=True
        )),
        "harness_sha256": digest(Path(__file__)),
    }


def validate_environment(config: dict) -> None:
    if platform.python_version() != config["environment"]["python"]:
        raise RuntimeError("Python differs from the H1 pin")
    for package in ("torch", "transformers", "flash-attn"):
        if version(package) != config["environment"][package]:
            raise RuntimeError(f"{package} differs from the H1 pin")


def write_result(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def model_manifest(config: dict) -> dict:
    """Check HF download metadata revision and hash the actual local files."""
    directory = ROOT / config["model"]["path"]
    files = ["config.json", "generation_config.json", "model.safetensors",
             "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "LICENSE"]
    manifest = {}
    for name in files:
        path = directory / name
        metadata = directory / ".cache/huggingface/download" / f"{name}.metadata"
        metadata_lines = metadata.read_text().splitlines()
        if metadata_lines[0] != config["model"]["revision"]:
            raise RuntimeError(f"Model revision mismatch: {name}")
        checksum = digest(path)
        if name.endswith(".safetensors") and checksum != metadata_lines[1]:
            raise RuntimeError(f"Weight SHA256 differs from the downloaded LFS object: {name}")
        manifest[name] = {"bytes": path.stat().st_size, "sha256": checksum}
    return {**config["model"], "files": manifest}

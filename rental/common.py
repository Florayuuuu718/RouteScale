"""Standard-library helpers; no GPU imports on the planning path."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def output(command, cwd=ROOT):
    return subprocess.check_output(list(map(str, command)), cwd=cwd, text=True, timeout=60).strip()


def relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe relative path: {name}')
    return path


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'Expected finite positive number: {value}')
    return value


def source_hashes(root=ROOT):
    # Only runtime source/configuration, not mutable result reports or personal files.
    names = output(['git', 'ls-files', '-z'], root).split('\0')
    prefixes = ('rental/', 'scripts/', 'serving/', 'config/', 'patches/', 'deploy/rental/')
    return {name: sha(root / name) for name in names if name and
            (name.startswith(prefixes) or ('/' not in name and name.endswith('.py')))}


def validate_hashes(root, hashes):
    if not hashes:
        raise ValueError('Empty evidence manifest')
    for name, digest in hashes.items():
        if sha(Path(root) / relative(name)) != digest:
            raise ValueError(f'Changed file: {name}')

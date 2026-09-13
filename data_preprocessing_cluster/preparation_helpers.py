"""Small I/O helpers preserved from the audited history-label builder."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def atomic_text(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(content)
        name = f.name
    os.replace(name, path)

def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def verify_files(root, hashes):
    root = Path(root).resolve()
    for name, digest in hashes.items():
        p = (root / name).resolve(strict=True)
        if root not in p.parents or sha256(p) != digest:
            raise ValueError(f"Artifact hash/path mismatch: {name}")

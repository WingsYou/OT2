#!/usr/bin/env python3
"""Verify the entire installed snapshot before tests, checks or inference."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath

MANIFEST = "configs/bundles/clinical_abcd_v1.manifest"
CHECKSUMS = "configs/bundles/clinical_abcd_v1.sha256"


def verify_bundle(root: Path) -> int:
    root = root.resolve(strict=True)
    names = (root / MANIFEST).read_text().splitlines()
    if not names or len(set(names)) != len(names):
        raise ValueError("empty or duplicate upload manifest")
    for name in names:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError(f"invalid manifest path: {name!r}")
        resolved = (root / name).resolve(strict=True)
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError(f"manifest entry is not a file inside the snapshot: {name}")
    if MANIFEST not in names or CHECKSUMS not in names:
        raise ValueError("manifest must include itself and checksum file")
    hashes = {}
    for line in (root / CHECKSUMS).read_text().splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest) or name in hashes:
            raise ValueError("invalid or duplicate checksum record")
        hashes[name] = digest
    if set(hashes) != set(names) - {CHECKSUMS}:
        raise ValueError("checksum coverage differs from upload manifest")
    for name, expected in hashes.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"snapshot hash mismatch: {name}; reinstall the complete upload bundle")
    return len(hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(f"SNAPSHOT_OK: clinical-abcd-v1; {verify_bundle(args.root)} files verified")


if __name__ == "__main__":
    main()

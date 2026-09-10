#!/usr/bin/env bash
# Verify, back up replaced code, and install this complete inference snapshot.
set -euo pipefail
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TARGET_DIR="${CODE_ROOT:-$HOME/ot2}"
MANIFEST="configs/bundles/basic_clinical_v1.manifest"
for name in python3 rsync; do
  command -v "$name" >/dev/null || { echo "Missing command: $name" >&2; exit 1; }
done
python3 "$SOURCE_DIR/verify_basic_clinical_bundle_v1.py" --root "$SOURCE_DIR"
mkdir -p -- "$TARGET_DIR"
TARGET_DIR="$(cd -- "$TARGET_DIR" && pwd -P)"
if [[ "$SOURCE_DIR" != "$TARGET_DIR" ]]; then
  BACKUP_ROOT="${BASIC_CODE_BACKUP_ROOT:-$HOME/ot2_code_backups}"
  mkdir -p -- "$BACKUP_ROOT"
  BACKUP_DIR="$(mktemp -d "$BACKUP_ROOT/basic_clinical_v1.XXXXXX")"
  echo "[install] Target: $TARGET_DIR"
  echo "[install] Replaced-file backup: $BACKUP_DIR"
  rsync -ac --backup --backup-dir="$BACKUP_DIR" \
    --files-from="$SOURCE_DIR/$MANIFEST" -- "$SOURCE_DIR/" "$TARGET_DIR/"
fi
python3 "$TARGET_DIR/verify_basic_clinical_bundle_v1.py" --root "$TARGET_DIR"
echo "UPLOAD_OK: basic-clinical-v1; code installed in $TARGET_DIR"

#!/usr/bin/env bash
# Install the verified modality-v4 coupled-ablation snapshot; preserve unrelated files.
set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TARGET_DIR="${CODE_ROOT:-$HOME/ot2}"
MANIFEST="configs/bundles/cluster_modality_v4.manifest"
CHECKSUMS="configs/bundles/cluster_modality_v4.sha256"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

for command_name in sha256sum rsync; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Required command missing: $command_name"
done
[[ -f "$SOURCE_DIR/$MANIFEST" ]] || fail "Upload manifest missing: $SOURCE_DIR/$MANIFEST"
[[ -f "$SOURCE_DIR/$CHECKSUMS" ]] || fail "Checksum file missing: $SOURCE_DIR/$CHECKSUMS"
[[ -s "$SOURCE_DIR/$MANIFEST" ]] || fail "Upload manifest is empty."
[[ -s "$SOURCE_DIR/$CHECKSUMS" ]] || fail "Checksum file is empty."

while IFS= read -r relative_path || [[ -n "$relative_path" ]]; do
  case "$relative_path" in
    ''|/*|..|../*|*/../*|*/..)
      fail "Invalid upload manifest path: $relative_path"
      ;;
  esac
  [[ -f "$SOURCE_DIR/$relative_path" ]] || fail "Snapshot file missing: $relative_path"
done < "$SOURCE_DIR/$MANIFEST"

printf '[install] Verifying source snapshot: %s\n' "$SOURCE_DIR"
(
  cd -- "$SOURCE_DIR"
  sha256sum --check --strict "$CHECKSUMS"
)

[[ ! -e "$TARGET_DIR" || -d "$TARGET_DIR" ]] || fail "Target is not a directory: $TARGET_DIR"
mkdir -p -- "$TARGET_DIR"
TARGET_DIR="$(cd -- "$TARGET_DIR" && pwd -P)"
if [[ "$SOURCE_DIR" != "$TARGET_DIR" ]]; then
  printf '[install] Updating code in: %s\n' "$TARGET_DIR"
  rsync -av --files-from="$SOURCE_DIR/$MANIFEST" -- "$SOURCE_DIR/" "$TARGET_DIR/"
else
  printf '[install] Snapshot is already in the target directory; skipping copy.\n'
fi

printf '[install] Verifying installed snapshot.\n'
(
  cd -- "$TARGET_DIR"
  sha256sum --check --strict "$CHECKSUMS"
)
printf 'UPLOAD_OK: modality-v4-coupled\n'

#!/usr/bin/env bash
# Validation-only universal masked-modality reconstruction pilot.
#
# This script reuses the already selected seed-42 full-assets checkpoints as
# frozen teachers.  It does not rebuild preprocessing and has no test action.
set -euo pipefail

STAGE="${1:-all}"
case "$STAGE" in
  all|cache|dry-run|train) ;;
  *)
    echo "usage: bash run_universal_reconstruction_v1.sh [all|cache|dry-run|train]" >&2
    exit 2
    ;;
esac

CODE_ROOT="${CODE_ROOT:-/home/pzhao3/ot2}"
OUT_ROOT="${OUT_ROOT:-/data/pzhao3/ot2_output}"
FULL_ROOT="${FULL_ROOT:-$OUT_ROOT/model_full_assets_v1}"
UNIVERSAL_ROOT="${UNIVERSAL_ROOT:-$OUT_ROOT/model_universal_reconstruction_v1}"
UNIVERSAL_CONFIG="${UNIVERSAL_CONFIG:-$CODE_ROOT/configs/universal_masking_v1.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Start with MoCA for quick feedback.  To run every current endpoint:
# TASKS="moca_binary cvd_broad_binary cvd_strict_binary" bash ... all
TASKS="${TASKS:-moca_binary}"
read -r -a TASK_LIST <<< "$TASKS"
if [[ "${#TASK_LIST[@]}" -eq 0 ]]; then
  echo "TASKS cannot be empty" >&2
  exit 2
fi

if [[ -r /home/pzhao3/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/pzhao3/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-ot2}"
fi

cd "$CODE_ROOT"

require_file() {
  if [[ ! -r "$1" ]]; then
    echo "required file is missing or unreadable: $1" >&2
    exit 1
  fi
}

config_for_task() {
  case "$1" in
    moca_binary) printf '%s\n' "$CODE_ROOT/configs/task2_v2_moca_full_assets_v1.yaml" ;;
    cvd_broad_binary) printf '%s\n' "$CODE_ROOT/configs/task2_v2_cvd_broad_full_assets_v1.yaml" ;;
    cvd_strict_binary) printf '%s\n' "$CODE_ROOT/configs/task2_v2_cvd_strict_full_assets_v1.yaml" ;;
    *) echo "unknown task in TASKS: $1" >&2; return 2 ;;
  esac
}

teacher_for_task() {
  printf '%s\n' "$FULL_ROOT/models/$1/seed42/best_model.pt"
}

cache_for_task() {
  printf '%s\n' "$UNIVERSAL_ROOT/token_cache/$1/seed42"
}

output_for_task() {
  printf '%s\n' "$UNIVERSAL_ROOT/models/$1/seed42"
}

check_gpu() {
  "$PYTHON_BIN" -c 'import sys, torch; ok=torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available()); print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False); sys.exit(0 if ok else 1)'
}

check_common() {
  require_file "$CODE_ROOT/train_universal_reconstruction_v2.py"
  require_file "$CODE_ROOT/task2_v2_universal_reconstruction.py"
  require_file "$CODE_ROOT/evaluate_universal_reconstruction_v2.py"
  require_file "$UNIVERSAL_CONFIG"
  "$PYTHON_BIN" -m py_compile \
    "$CODE_ROOT/task2_v2_universal_reconstruction.py" \
    "$CODE_ROOT/train_universal_reconstruction_v2.py" \
    "$CODE_ROOT/evaluate_universal_reconstruction_v2.py"
  check_gpu
}

run_cache() {
  check_common
  local task config teacher cache_dir log_dir cache_policy_args
  for task in "${TASK_LIST[@]}"; do
    config="$(config_for_task "$task")"
    teacher="$(teacher_for_task "$task")"
    cache_dir="$(cache_for_task "$task")"
    log_dir="$UNIVERSAL_ROOT/logs/$task/seed42"
    require_file "$config"
    require_file "$teacher"
    mkdir -p "$cache_dir" "$log_dir"
    cache_policy_args=(--reuse-existing)
    if [[ "${UNIVERSAL_FORCE_RECACHE:-0}" == "1" ]]; then
      cache_policy_args=(--overwrite)
    fi
    echo "[cache][$task] Encoding train and validation once; test is not iterated"
    "$PYTHON_BIN" train_universal_reconstruction_v2.py cache \
      --config "$config" \
      --teacher-checkpoint "$teacher" \
      --cache-dir "$cache_dir" \
      --seed 42 \
      --device cuda \
      --workers "${UNIVERSAL_CACHE_WORKERS:-2}" \
      --batch-size "${UNIVERSAL_CACHE_BATCH_SIZE:-2}" \
      --bf16 \
      "${cache_policy_args[@]}" \
      2>&1 | tee "$log_dir/01_token_cache.log"
    require_file "$cache_dir/${task}_train_teacher_tokens.pt"
    require_file "$cache_dir/${task}_val_teacher_tokens.pt"
    require_file "$cache_dir/${task}_token_cache_summary.json"
  done
}

run_dry_run() {
  check_common
  local task config teacher cache_dir output_dir log_dir
  for task in "${TASK_LIST[@]}"; do
    config="$(config_for_task "$task")"
    teacher="$(teacher_for_task "$task")"
    cache_dir="$(cache_for_task "$task")"
    output_dir="$(output_for_task "$task")"
    log_dir="$UNIVERSAL_ROOT/logs/$task/seed42"
    require_file "$config"
    require_file "$teacher"
    require_file "$cache_dir/${task}_train_teacher_tokens.pt"
    require_file "$cache_dir/${task}_val_teacher_tokens.pt"
    mkdir -p "$output_dir" "$log_dir"
    echo "[dry-run][$task] One token batch, backward pass, no epoch and no test"
    "$PYTHON_BIN" train_universal_reconstruction_v2.py train \
      --config "$config" \
      --universal-config "$UNIVERSAL_CONFIG" \
      --teacher-checkpoint "$teacher" \
      --cache-dir "$cache_dir" \
      --output-dir "$output_dir" \
      --seed 42 \
      --device cuda \
      --bf16 \
      --dry-run \
      2>&1 | tee "$log_dir/02_dry_run.log"
    require_file "$output_dir/dry_run_summary.json"
  done
}

completed_run_is_current() {
  local task="$1"
  local output_dir="$2"
  local config="$3"
  local teacher="$4"
  [[ -r "$output_dir/run_summary.json" && -r "$output_dir/deployment_universal_student.pt" ]] || return 1
  "$PYTHON_BIN" - \
    "$output_dir/run_summary.json" \
    "$task" \
    "$config" \
    "$teacher" \
    "$UNIVERSAL_CONFIG" \
    "$output_dir/deployment_universal_student.pt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
provenance = summary.get("provenance", {})
# Resolve current runtime code from the explicit supervised-config location.
code_root = Path(sys.argv[3]).resolve().parents[1]
runtime_names = (
    "train_universal_reconstruction_v2.py",
    "task2_v2_universal_reconstruction.py",
    "task2_v2_model.py",
    "task2_v2_distill.py",
    "task2_v2_engine.py",
    "task2_v2_config.py",
    "task2_v2_dataset.py",
    "train_task2_v2.py",
    "train_cheap_student_v2.py",
    "fit_train_scalers_v2.py",
    "preprocessing_v2.py",
)
current_runtime = {
    name: sha256(code_root / name)
    for name in runtime_names
    if (code_root / name).is_file()
}
deployment_path = Path(sys.argv[6]).resolve()
deployment_record = summary.get("artifacts", {}).get("deployment_checkpoint")
deployment_valid = (
    isinstance(deployment_record, dict)
    and Path(str(deployment_record.get("path", ""))).resolve() == deployment_path
    and deployment_path.is_file()
    and deployment_path.stat().st_size > 0
    and deployment_record.get("size_bytes") == deployment_path.stat().st_size
    and deployment_record.get("sha256") == sha256(deployment_path)
)
valid = (
    summary.get("format_version") == "ot2-universal-reconstruction-training-v1"
    and summary.get("status") == "complete"
    and summary.get("task") == sys.argv[2]
    and summary.get("seed") == 42
    and summary.get("test_evaluated") is False
    and provenance.get("supervised_config", {}).get("sha256") == sha256(sys.argv[3])
    and provenance.get("teacher_checkpoint", {}).get("sha256") == sha256(sys.argv[4])
    and provenance.get("universal_config", {}).get("sha256") == sha256(sys.argv[5])
    and len(current_runtime) == len(runtime_names)
    and provenance.get("runtime_code_sha256") == current_runtime
    and deployment_valid
)
raise SystemExit(0 if valid else 1)
PY
}

run_train() {
  check_common
  local task config teacher cache_dir output_dir log_dir overwrite_args
  for task in "${TASK_LIST[@]}"; do
    config="$(config_for_task "$task")"
    teacher="$(teacher_for_task "$task")"
    cache_dir="$(cache_for_task "$task")"
    output_dir="$(output_for_task "$task")"
    log_dir="$UNIVERSAL_ROOT/logs/$task/seed42"
    require_file "$config"
    require_file "$teacher"
    require_file "$cache_dir/${task}_train_teacher_tokens.pt"
    require_file "$cache_dir/${task}_val_teacher_tokens.pt"
    mkdir -p "$output_dir" "$log_dir"

    if [[ "${UNIVERSAL_FORCE_RETRAIN:-0}" != "1" ]] && completed_run_is_current "$task" "$output_dir" "$config" "$teacher"; then
      echo "[train][$task] Current completed validation-only run found; skipping"
      continue
    fi
    overwrite_args=()
    if [[ "${UNIVERSAL_FORCE_RETRAIN:-0}" == "1" ]]; then
      overwrite_args+=(--overwrite-output)
    fi
    echo "[train][$task] Training in cached token space; selection uses validation only"
    "$PYTHON_BIN" train_universal_reconstruction_v2.py train \
      --config "$config" \
      --universal-config "$UNIVERSAL_CONFIG" \
      --teacher-checkpoint "$teacher" \
      --cache-dir "$cache_dir" \
      --output-dir "$output_dir" \
      --seed 42 \
      --device cuda \
      --bf16 \
      "${overwrite_args[@]}" \
      2>&1 | tee "$log_dir/03_train.log"
    if ! completed_run_is_current "$task" "$output_dir" "$config" "$teacher"; then
      echo "[train][$task] output failed completion/provenance check" >&2
      exit 1
    fi
  done
}

case "$STAGE" in
  all)
    run_cache
    run_dry_run
    run_train
    ;;
  cache) run_cache ;;
  dry-run) run_dry_run ;;
  train) run_train ;;
esac

echo "Universal reconstruction V1 stage complete: $STAGE"
echo "Validation-only outputs: $UNIVERSAL_ROOT/models"

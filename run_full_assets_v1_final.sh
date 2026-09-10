#!/usr/bin/env bash
# Freeze and evaluate the already-trained full-assets V1 models on test.
# Test access is a separate, explicit stage from validation training.
set -euo pipefail

STAGE="${1:-all}"
case "$STAGE" in
  all|lock|test|summary) ;;
  *)
    echo "usage: bash run_full_assets_v1_final.sh [all|lock|test|summary]" >&2
    exit 2
    ;;
esac

CODE_ROOT="${CODE_ROOT:-/home/pzhao3/ot2}"
OUT_ROOT="${OUT_ROOT:-/data/pzhao3/ot2_output}"
FULL_ROOT="${FULL_ROOT:-$OUT_ROOT/model_full_assets_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_OUT_ROOT="/data/pzhao3/ot2_output"
EXPECTED_FULL_ROOT="$EXPECTED_OUT_ROOT/model_full_assets_v1"

if [[ "$OUT_ROOT" != "$EXPECTED_OUT_ROOT" || "$FULL_ROOT" != "$EXPECTED_FULL_ROOT" ]]; then
  echo "This V1 bundle is path-bound to OUT_ROOT=$EXPECTED_OUT_ROOT" >&2
  echo "Received OUT_ROOT=$OUT_ROOT and FULL_ROOT=$FULL_ROOT" >&2
  exit 2
fi

if [[ -r /home/pzhao3/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/pzhao3/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-ot2}"
fi

cd "$CODE_ROOT"

TASKS=(moca_binary cvd_broad_binary cvd_strict_binary)
RETINAL_SUMMARY="$FULL_ROOT/retinal_model_view/retinal_model_view_summary.json"
FINAL_ROOT="$FULL_ROOT/final_test"
SUMMARY_ROOT="$FULL_ROOT/benchmark_summary"
LOG_DIR="$FULL_ROOT/logs"

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
    *) echo "unknown task: $1" >&2; return 2 ;;
  esac
}

check_gpu() {
  "$PYTHON_BIN" -c 'import sys, torch; print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available()); print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); sys.exit(0 if torch.cuda.is_available() else 1)'
}

validate_existing_lock() {
  local config="$1"
  local checkpoint="$2"
  local output="$3"
  local lock="$4"
  "$PYTHON_BIN" - "$lock" "$config" "$checkpoint" "$output" <<'PY'
import sys
from freeze_supervised_final_lock_v2 import validate_supervised_final_lock_preflight

lock, config, checkpoint, output = sys.argv[1:]
audit = validate_supervised_final_lock_preflight(
    lock,
    config_path=config,
    checkpoint_path=checkpoint,
    output_path=output,
    model_id="full_teacher",
    model_role="full_teacher",
    seed=42,
)
print("existing immutable final lock: PASS", audit)
PY
}

validate_existing_test() {
  local task="$1"
  local config="$2"
  local checkpoint="$3"
  local output="$4"
  local lock="$5"
  "$PYTHON_BIN" - "$task" "$config" "$checkpoint" "$output" "$lock" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

task, config_raw, checkpoint_raw, output_raw, lock_raw = sys.argv[1:]
config = Path(config_raw).resolve()
checkpoint = Path(checkpoint_raw).resolve()
output = Path(output_raw).resolve()
metadata_path = output.with_suffix(output.suffix + ".metadata.json")
lock = Path(lock_raw).resolve()

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in (config, checkpoint, output, metadata_path, lock):
    if not path.is_file():
        raise FileNotFoundError(path)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
assert metadata["schema_version"] == "ot2-frozen-evaluation-v2-1", metadata
assert metadata["analysis_role"] == "locked_confirmatory_test", metadata
assert metadata["task"] == task and metadata["split"] == "test", metadata
assert metadata["model_id"] == "full_teacher", metadata
assert metadata["model_role"] == "full_teacher", metadata
assert int(metadata["seed"]) == 42, metadata
assert metadata["predictions"]["path"] == str(output), metadata
assert metadata["predictions"]["sha256"] == sha(output), metadata
sources = metadata["source_files"]
for name, path in (("config", config), ("checkpoint", checkpoint)):
    assert sources[name]["path"] == str(path), (name, sources[name], path)
    assert sources[name]["sha256"] == sha(path), (name, sources[name])
lock_audit = metadata["selection_lock"]
assert lock_audit["path"] == str(lock), lock_audit
assert lock_audit["sha256"] == sha(lock), lock_audit
assert lock_audit["schema_version"] == "ot2-supervised-final-lock-v2-1", lock_audit
assert float(metadata["threshold_from_validation"]) == float(
    lock_audit["validation_threshold"]
), metadata
print("existing locked test evidence: PASS", task, metadata["metrics"])
PY
}

run_lock() {
  require_file "$RETINAL_SUMMARY"
  mkdir -p "$FINAL_ROOT" "$LOG_DIR"

  local task config run_dir checkpoint run_summary task_final output metadata lock
  for task in "${TASKS[@]}"; do
    config="$(config_for_task "$task")"
    run_dir="$FULL_ROOT/models/$task/seed42"
    checkpoint="$run_dir/best_model.pt"
    run_summary="$run_dir/run_summary.json"
    task_final="$FINAL_ROOT/$task"
    output="$task_final/test_predictions.csv"
    metadata="$output.metadata.json"
    lock="$task_final/supervised_final_lock.json"
    require_file "$config"
    require_file "$checkpoint"
    require_file "$run_summary"
    mkdir -p "$task_final"

    if [[ -e "$lock" ]]; then
      if [[ -r "$output" && -r "$metadata" ]]; then
        validate_existing_test "$task" "$config" "$checkpoint" "$output" "$lock"
      elif [[ ! -e "$output" && ! -e "$metadata" ]]; then
        validate_existing_lock "$config" "$checkpoint" "$output" "$lock"
      else
        echo "partial final-test evidence exists for $task; refusing to continue" >&2
        exit 1
      fi
      continue
    fi
    if [[ -e "$output" || -e "$metadata" ]]; then
      echo "test output exists without its immutable lock for $task" >&2
      exit 1
    fi

    echo "[lock][$task] Freezing seed42 checkpoint and validation threshold"
    "$PYTHON_BIN" freeze_supervised_final_lock_v2.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --run-summary "$run_summary" \
      --retinal-sidecar-summary "$RETINAL_SUMMARY" \
      --authorized-output "$output" \
      --out "$lock" \
      2>&1 | tee "$LOG_DIR/07_lock_${task}.log"
    require_file "$lock"
    validate_existing_lock "$config" "$checkpoint" "$output" "$lock"
  done
}

run_test() {
  mkdir -p "$FINAL_ROOT" "$LOG_DIR"
  check_gpu

  local amp_flag="--no-bf16"
  if "$PYTHON_BIN" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    amp_flag="--bf16"
  fi
  echo "[test] AMP choice: $amp_flag"

  local task config run_dir checkpoint task_final output metadata lock
  for task in "${TASKS[@]}"; do
    config="$(config_for_task "$task")"
    run_dir="$FULL_ROOT/models/$task/seed42"
    checkpoint="$run_dir/best_model.pt"
    task_final="$FINAL_ROOT/$task"
    output="$task_final/test_predictions.csv"
    metadata="$output.metadata.json"
    lock="$task_final/supervised_final_lock.json"
    require_file "$config"
    require_file "$checkpoint"
    require_file "$lock"

    if [[ -r "$output" && -r "$metadata" ]]; then
      validate_existing_test "$task" "$config" "$checkpoint" "$output" "$lock"
      echo "[test][$task] Current locked evidence found; skipping evaluation"
      continue
    fi
    if [[ -e "$output" || -e "$metadata" ]]; then
      echo "partial final-test evidence exists for $task; refusing to continue" >&2
      exit 1
    fi

    echo "[test][$task] One-time frozen checkpoint evaluation"
    "$PYTHON_BIN" evaluate_frozen_checkpoint_v2.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split test \
      --model-id full_teacher \
      --model-role full_teacher \
      --seed 42 \
      --selection-lock "$lock" \
      --out "$output" \
      --device cuda \
      --batch-size "${FULL_EVAL_BATCH_SIZE:-2}" \
      --workers "${FULL_DATALOADER_WORKERS:-2}" \
      "$amp_flag" \
      2>&1 | tee "$LOG_DIR/08_test_${task}.log"
    require_file "$output"
    require_file "$metadata"
    validate_existing_test "$task" "$config" "$checkpoint" "$output" "$lock"
  done
}

run_summary() {
  local task
  for task in "${TASKS[@]}"; do
    require_file "$FULL_ROOT/models/$task/seed42/run_summary.json"
    require_file "$FINAL_ROOT/$task/test_predictions.csv"
    require_file "$FINAL_ROOT/$task/test_predictions.csv.metadata.json"
  done
  mkdir -p "$SUMMARY_ROOT" "$LOG_DIR"
  "$PYTHON_BIN" summarize_three_task_benchmark_v1.py \
    --validation "moca_binary=$FULL_ROOT/models/moca_binary/seed42" \
    --validation "cvd_broad_binary=$FULL_ROOT/models/cvd_broad_binary/seed42" \
    --validation "cvd_strict_binary=$FULL_ROOT/models/cvd_strict_binary/seed42" \
    --test "moca_binary=$FINAL_ROOT/moca_binary/test_predictions.csv" \
    --test "cvd_broad_binary=$FINAL_ROOT/cvd_broad_binary/test_predictions.csv" \
    --test "cvd_strict_binary=$FINAL_ROOT/cvd_strict_binary/test_predictions.csv" \
    --out-dir "$SUMMARY_ROOT" \
    --benchmark-name "OT2 V2 full-curated-retinal three-task benchmark (seed 42)" \
    2>&1 | tee "$LOG_DIR/09_three_task_summary.log"
  require_file "$SUMMARY_ROOT/three_task_benchmark_v1.md"
  cat "$SUMMARY_ROOT/three_task_benchmark_v1.md"
}

case "$STAGE" in
  all)
    run_lock
    run_test
    run_summary
    ;;
  lock) run_lock ;;
  test) run_test ;;
  summary) run_summary ;;
esac

echo "Full-assets V1 final stage complete: $STAGE"
echo "Report: $SUMMARY_ROOT/three_task_benchmark_v1.md"

#!/usr/bin/env bash
# Validation-screen experiment for all three downstream tasks using every
# successful canonical retinal asset (60,807).  This runner deliberately has
# no final-test stage; checkpoint locking and one-time test evaluation happen
# only after the validation design is frozen.
set -euo pipefail

STAGE="${1:-all}"
case "$STAGE" in
  all|prepare|dry-run|train) ;;
  *)
    echo "usage: bash run_full_assets_v1.sh [all|prepare|dry-run|train]" >&2
    exit 2
    ;;
esac

CODE_ROOT="${CODE_ROOT:-/home/pzhao3/ot2}"
DATASET_ROOT="${DATASET_ROOT:-/data/EMED/AIREADI/aireadi300/d55855bd-e2fb-4e53-a07f-e5fd3c41707a/dataset}"
OUT_ROOT="${OUT_ROOT:-/data/pzhao3/ot2_output}"
FULL_ROOT="${FULL_ROOT:-$OUT_ROOT/model_full_assets_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# The three checked-in YAML files intentionally contain absolute cluster paths.
# Fail before writing anything if an override would make those paths disagree.
EXPECTED_OUT_ROOT="/data/pzhao3/ot2_output"
EXPECTED_FULL_ROOT="$EXPECTED_OUT_ROOT/model_full_assets_v1"
if [[ "$OUT_ROOT" != "$EXPECTED_OUT_ROOT" || "$FULL_ROOT" != "$EXPECTED_FULL_ROOT" ]]; then
  echo "This V1 bundle is path-bound to OUT_ROOT=$EXPECTED_OUT_ROOT" >&2
  echo "Received OUT_ROOT=$OUT_ROOT and FULL_ROOT=$FULL_ROOT" >&2
  echo "Create a new set of path-bound YAML configs before changing these roots." >&2
  exit 2
fi

if [[ -r /home/pzhao3/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/pzhao3/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-ot2}"
fi

cd "$CODE_ROOT"

TASKS=(moca_binary cvd_broad_binary cvd_strict_binary)
RETINAL_CANONICAL_INDEX="$OUT_ROOT/retinal/retinal_preprocessed_index.csv"
RETINAL_SIDECAR_DIR="$FULL_ROOT/retinal_model_view"
RETINAL_SIDECAR_INDEX="$RETINAL_SIDECAR_DIR/retinal_model_view_index.csv"
RETINAL_SIDECAR_SUMMARY="$RETINAL_SIDECAR_DIR/retinal_model_view_summary.json"
INPUT_DIR="$FULL_ROOT/inputs"
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

clinical_wide_for_task() {
  local task="$1"
  local wide="$OUT_ROOT/clinical/$task/clinical_features_${task}_wide.csv"
  local legacy="$OUT_ROOT/clinical/$task/clinical_features_${task}.csv"
  if [[ -r "$wide" ]]; then
    printf '%s\n' "$wide"
  elif [[ -r "$legacy" ]]; then
    printf '%s\n' "$legacy"
  else
    echo "clinical feature file is missing for $task: checked $wide and $legacy" >&2
    return 1
  fi
}

check_gpu() {
  "$PYTHON_BIN" -c 'import sys, torch; available=torch.cuda.is_available(); bf16=bool(available and torch.cuda.is_bf16_supported()); print("torch:", torch.__version__); print("cuda:", available); print("gpu:", torch.cuda.get_device_name(0) if available else "NONE"); print("bf16:", bf16); sys.exit(0 if available and bf16 else 1)'
}

validate_full_retinal_sidecars() {
  "$PYTHON_BIN" - \
    "$RETINAL_CANONICAL_INDEX" \
    "$RETINAL_SIDECAR_INDEX" \
    "$RETINAL_SIDECAR_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

canonical_path, sidecar_path, summary_path = map(Path, sys.argv[1:])
expected_rows = 60_807
expected_caps = {"photography": 6, "oct": 2, "octa": 16, "flio": 4}
success_statuses = {
    "processed", "cached", "skipped", "skipped_current",
    "processed_with_source_errors",
}

summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["model_view_config"]["selection_caps_per_person_modality"] == expected_caps, summary
assert summary["n_successful_input_rows"] == expected_rows, summary
assert summary["n_selected"] == expected_rows, summary
assert summary["n_omitted_by_cap"] == 0, summary
assert summary["n_failed"] == 0, summary
assert summary["n_processed"] + summary["n_cached"] == expected_rows, summary

canonical = pd.read_csv(
    canonical_path,
    dtype={"person_id": "string"},
    keep_default_na=False,
)
status = canonical["status"].astype(str).str.strip().str.lower()
canonical = canonical.loc[status.isin(success_statuses)].copy()
canonical["modality"] = canonical["modality"].astype(str).str.strip().str.lower()
assert len(canonical) == expected_rows, len(canonical)
assert not canonical["asset_key"].duplicated().any()

observed_maxima = {
    str(modality): int(value)
    for modality, value in (
        canonical.groupby(["person_id", "modality"], sort=False)
        .size()
        .groupby(level="modality")
        .max()
        .items()
    )
}
assert observed_maxima == expected_caps, observed_maxima

sidecar = pd.read_csv(
    sidecar_path,
    dtype={"person_id": "string"},
    keep_default_na=False,
)
assert len(sidecar) == expected_rows, len(sidecar)
assert not sidecar["asset_key"].duplicated().any()
assert set(sidecar["asset_key"]) == set(canonical["asset_key"])
assert set(sidecar["model_view_config_hash"]) == {
    summary["model_view_config_hash"]
}
sidecar_status = sidecar["status"].astype(str).str.strip().str.lower()
assert sidecar_status.isin({"processed", "cached"}).all()

print(
    "full retinal sidecar check: PASS",
    {
        "assets": len(sidecar),
        "omitted": summary["n_omitted_by_cap"],
        "failures": summary["n_failed"],
        "observed_per_person_maxima": observed_maxima,
    },
)
PY
}

validate_task_config() {
  local task="$1"
  local config="$2"
  "$PYTHON_BIN" - "$config" "$task" <<'PY'
import sys

from task2_v2_config import PRIMARY_MODALITIES, load_model_v2_config

config_path, expected_task = sys.argv[1:]
config = load_model_v2_config(config_path, validate_files=True)
expected_caps = {"photography": 6, "oct": 2, "octa": 16, "flio": 4}
assert config.dataset.task == expected_task
assert dict(config.dataset.retinal_max_assets) == expected_caps
assert list(config.model.source_order) == list(PRIMARY_MODALITIES)
training = config.raw["training"]
assert training["evaluation_phase"] == "validation_screen"
assert config.training.batch_size == 2
assert config.training.gradient_accumulation_steps == 16
assert config.training.max_epochs == 30
assert config.training.early_stopping_patience == 7
assert training["num_workers"] == 2
assert training["bf16"] is True
assert config.importance.allow_test_set_model_selection is False
print("strict config/lineage check: PASS", expected_task, config.config_hash)
PY
}

run_prepare() {
  require_file "$DATASET_ROOT/participants.tsv"
  require_file "$RETINAL_CANONICAL_INDEX"
  require_file "$OUT_ROOT/environment_v4/environment_5min_index.csv"
  require_file "$OUT_ROOT/wearable_v5/wearable_5min_index.csv"
  require_file "$OUT_ROOT/ecg/ecg_index.csv"
  mkdir -p "$RETINAL_SIDECAR_DIR" "$INPUT_DIR" "$LOG_DIR"

  echo "[prepare] Building/reusing sidecars for all 60,807 canonical retinal assets"
  "$PYTHON_BIN" build_retinal_model_view_sidecars_v2.py \
    --retinal-index "$RETINAL_CANONICAL_INDEX" \
    --out-dir "$RETINAL_SIDECAR_DIR" \
    --num-workers "${FULL_SIDECAR_WORKERS:-128}" \
    --cache-policy reuse \
    --max-photography-assets 6 \
    --max-oct-assets 2 \
    --max-octa-assets 16 \
    --max-flio-assets 4 \
    2>&1 | tee "$LOG_DIR/01_retinal_sidecars.log"

  require_file "$RETINAL_SIDECAR_INDEX"
  require_file "$RETINAL_SIDECAR_SUMMARY"
  validate_full_retinal_sidecars \
    2>&1 | tee "$LOG_DIR/02_validate_full_retinal.log"

  local task config clinical_wide clinical_schema master_csv master_metadata
  for task in "${TASKS[@]}"; do
    config="$(config_for_task "$task")"
    clinical_wide="$(clinical_wide_for_task "$task")"
    clinical_schema="$OUT_ROOT/clinical/$task/clinical_schema_${task}.json"
    master_csv="$INPUT_DIR/master_${task}.csv"
    master_metadata="$INPUT_DIR/master_${task}.metadata.json"

    require_file "$config"
    require_file "$OUT_ROOT/labels/labels_${task}.csv"
    require_file "$clinical_wide"
    require_file "$clinical_schema"
    require_file "$OUT_ROOT/scalers/train_scalers_${task}.json"

    echo "[prepare][$task] Rebuilding strict multimodal master index"
    "$PYTHON_BIN" build_multimodal_index_v2.py \
      --participants "$DATASET_ROOT/participants.tsv" \
      --labels "$OUT_ROOT/labels/labels_${task}.csv" \
      --clinical_features "$clinical_wide" \
      --clinical_schema "$clinical_schema" \
      --retinal_index "$RETINAL_SIDECAR_INDEX" \
      --environment_index "$OUT_ROOT/environment_v4/environment_5min_index.csv" \
      --wearable_index "$OUT_ROOT/wearable_v5/wearable_5min_index.csv" \
      --ecg_index "$OUT_ROOT/ecg/ecg_index.csv" \
      --artifact_root "$OUT_ROOT" \
      --out_csv "$master_csv" \
      --metadata_json "$master_metadata" \
      2>&1 | tee "$LOG_DIR/03_master_${task}.log"

    require_file "$master_csv"
    require_file "$master_metadata"
    validate_task_config "$task" "$config" \
      2>&1 | tee "$LOG_DIR/04_config_${task}.log"
  done
}

run_dry_run() {
  mkdir -p "$LOG_DIR"
  check_gpu
  echo "[dry-run] AMP choice: --bf16"

  local task config
  for task in "${TASKS[@]}"; do
    config="$(config_for_task "$task")"
    require_file "$config"
    validate_task_config "$task" "$config"
    echo "[dry-run][$task] Loading one train batch; val/test loaders stay unopened"
    "$PYTHON_BIN" train_task2_v2.py \
      --config "$config" \
      --device cuda \
      --workers "${FULL_DATALOADER_WORKERS:-2}" \
      --bf16 \
      --evaluation-phase validation_screen \
      --dry-run \
      2>&1 | tee "$LOG_DIR/05_dry_run_${task}.log"
  done
}

completed_run_is_current() {
  local task="$1"
  local config="$2"
  local run_dir="$FULL_ROOT/models/$task/seed42"
  [[ -r "$run_dir/run_summary.json" && -r "$run_dir/best_model.pt" ]] || return 1
  "$PYTHON_BIN" - "$config" "$run_dir/run_summary.json" <<'PY'
import json
import sys

from task2_v2_config import load_model_v2_config

config = load_model_v2_config(sys.argv[1], validate_files=True)
summary = json.loads(open(sys.argv[2], encoding="utf-8").read())
valid = (
    summary.get("task") == config.dataset.task
    and summary.get("evaluation_phase") == "validation_screen"
    and summary.get("test_evaluated") is False
    and summary.get("config_provenance", {}).get("config_hash") == config.config_hash
)
raise SystemExit(0 if valid else 1)
PY
}

validate_completed_run() {
  local task="$1"
  local config="$2"
  local run_dir="$FULL_ROOT/models/$task/seed42"
  require_file "$run_dir/best_model.pt"
  require_file "$run_dir/run_summary.json"
  if ! completed_run_is_current "$task" "$config"; then
    echo "completed output failed task/config/validation-only provenance check: $run_dir" >&2
    exit 1
  fi
  echo "[train][$task] Completed checkpoint provenance: PASS"
}

run_train() {
  mkdir -p "$LOG_DIR"
  check_gpu
  echo "[train] AMP choice: --bf16"

  local task config run_dir
  for task in "${TASKS[@]}"; do
    config="$(config_for_task "$task")"
    run_dir="$FULL_ROOT/models/$task/seed42"
    require_file "$config"
    validate_task_config "$task" "$config"

    if [[ "${FULL_FORCE_RETRAIN:-0}" != "1" ]] && completed_run_is_current "$task" "$config"; then
      echo "[train][$task] Current completed run found; skipping"
      continue
    fi
    if [[ -r "$run_dir/run_summary.json" && "${FULL_FORCE_RETRAIN:-0}" != "1" ]]; then
      echo "[train][$task] Existing completed output does not match this config." >&2
      echo "Set FULL_FORCE_RETRAIN=1 only if replacing it is intended." >&2
      exit 1
    fi
    if [[ -d "$run_dir" && ! -r "$run_dir/run_summary.json" ]]; then
      echo "[train][$task] Partial run found; this engine restarts the task from epoch 1"
    fi
    mkdir -p "$run_dir"
    echo "[train][$task] Starting validation-screen training"
    "$PYTHON_BIN" train_task2_v2.py \
      --config "$config" \
      --device cuda \
      --workers "${FULL_DATALOADER_WORKERS:-2}" \
      --bf16 \
      --evaluation-phase validation_screen \
      2>&1 | tee "$LOG_DIR/06_train_${task}.log"
    validate_completed_run "$task" "$config"
  done
}

case "$STAGE" in
  all)
    run_prepare
    run_dry_run
    run_train
    ;;
  prepare) run_prepare ;;
  dry-run) run_dry_run ;;
  train) run_train ;;
esac

echo "Full-assets V1 stage complete: $STAGE"
echo "Validation-only outputs: $FULL_ROOT/models"

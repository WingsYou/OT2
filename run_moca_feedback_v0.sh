#!/usr/bin/env bash
# Preliminary MoCA feedback run: prepare compact retinal views, train one
# nine-source model, and evaluate frozen validation ablations.
set -euo pipefail

STAGE="${1:-all}"
case "$STAGE" in
  all|prepare|train|importance|summary) ;;
  *)
    echo "usage: bash run_moca_feedback_v0.sh [all|prepare|train|importance|summary]" >&2
    exit 2
    ;;
esac

CODE_ROOT="${CODE_ROOT:-/home/pzhao3/ot2}"
DATASET_ROOT="${DATASET_ROOT:-/data/EMED/AIREADI/aireadi300/d55855bd-e2fb-4e53-a07f-e5fd3c41707a/dataset}"
OUT_ROOT="${OUT_ROOT:-/data/pzhao3/ot2_output}"
V0_ROOT="${V0_ROOT:-$OUT_ROOT/model_v0_feedback}"
CONFIG="${V0_CONFIG:-$CODE_ROOT/configs/task2_v2_moca_feedback_v0.yaml}"

if [[ -r /home/pzhao3/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/pzhao3/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-ot2}"
fi

cd "$CODE_ROOT"

RETINAL_CANONICAL_INDEX="$OUT_ROOT/retinal/retinal_preprocessed_index.csv"
RETINAL_SIDECAR_DIR="$V0_ROOT/retinal_model_view"
RETINAL_SIDECAR_INDEX="$RETINAL_SIDECAR_DIR/retinal_model_view_index.csv"
INPUT_DIR="$V0_ROOT/inputs"
MASTER_CSV="$INPUT_DIR/master_moca_binary_sidecar.csv"
MASTER_METADATA="$INPUT_DIR/master_moca_binary_sidecar.metadata.json"
RUN_DIR="$V0_ROOT/moca_seed42"
IMPORTANCE_DIR="$RUN_DIR/importance_val_ablation"

CLINICAL_SCHEMA="$OUT_ROOT/clinical/moca_binary/clinical_schema_moca_binary.json"
if [[ -n "${CLINICAL_WIDE:-}" ]]; then
  :
elif [[ -r "$OUT_ROOT/clinical/moca_binary/clinical_features_moca_binary_wide.csv" ]]; then
  CLINICAL_WIDE="$OUT_ROOT/clinical/moca_binary/clinical_features_moca_binary_wide.csv"
else
  CLINICAL_WIDE="$OUT_ROOT/clinical/moca_binary/clinical_features_moca_binary.csv"
fi

require_file() {
  if [[ ! -r "$1" ]]; then
    echo "required file is missing or unreadable: $1" >&2
    exit 1
  fi
}

check_gpu() {
  python -c 'import sys, torch; print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available()); print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); sys.exit(0 if torch.cuda.is_available() else 1)'
}

run_prepare() {
  require_file "$DATASET_ROOT/participants.tsv"
  require_file "$OUT_ROOT/labels/labels_moca_binary.csv"
  require_file "$CLINICAL_WIDE"
  require_file "$CLINICAL_SCHEMA"
  require_file "$RETINAL_CANONICAL_INDEX"
  require_file "$OUT_ROOT/environment_v4/environment_5min_index.csv"
  require_file "$OUT_ROOT/wearable_v5/wearable_5min_index.csv"
  require_file "$OUT_ROOT/ecg/ecg_index.csv"
  mkdir -p "$RETINAL_SIDECAR_DIR" "$INPUT_DIR" "$RUN_DIR/logs"

  python build_retinal_model_view_sidecars_v2.py \
    --retinal-index "$RETINAL_CANONICAL_INDEX" \
    --out-dir "$RETINAL_SIDECAR_DIR" \
    --num-workers "${V0_SIDECAR_WORKERS:-128}" \
    --cache-policy reuse \
    --max-photography-assets 2 \
    --max-oct-assets 1 \
    --max-octa-assets 2 \
    --max-flio-assets 1 \
    2>&1 | tee "$RUN_DIR/logs/01_retinal_sidecars.log"

  require_file "$RETINAL_SIDECAR_INDEX"
  python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); x=json.loads(p.read_text()); assert x.get("n_failed", 0)==0, x; assert x.get("n_selected", 0)==x.get("n_processed", 0)+x.get("n_cached", 0), x; print("retinal sidecar check: PASS", {k:x.get(k) for k in ("n_selected","n_processed","n_cached","n_failed")})' \
    "$RETINAL_SIDECAR_DIR/retinal_model_view_summary.json"

  python build_multimodal_index_v2.py \
    --participants "$DATASET_ROOT/participants.tsv" \
    --labels "$OUT_ROOT/labels/labels_moca_binary.csv" \
    --clinical_features "$CLINICAL_WIDE" \
    --clinical_schema "$CLINICAL_SCHEMA" \
    --retinal_index "$RETINAL_SIDECAR_INDEX" \
    --environment_index "$OUT_ROOT/environment_v4/environment_5min_index.csv" \
    --wearable_index "$OUT_ROOT/wearable_v5/wearable_5min_index.csv" \
    --ecg_index "$OUT_ROOT/ecg/ecg_index.csv" \
    --artifact_root "$OUT_ROOT" \
    --out_csv "$MASTER_CSV" \
    --metadata_json "$MASTER_METADATA" \
    2>&1 | tee "$RUN_DIR/logs/02_master_index.log"

  require_file "$CONFIG"
  python -c 'import sys; from task2_v2_config import load_model_v2_config; c=load_model_v2_config(sys.argv[1], validate_files=True); print("V0 lineage check: PASS", c.dataset.task, c.config_hash)' "$CONFIG"
}

run_train() {
  require_file "$CONFIG"
  require_file "$MASTER_CSV"
  require_file "$MASTER_METADATA"
  mkdir -p "$RUN_DIR/logs"
  check_gpu

  local amp_flag="--no-bf16"
  if python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    amp_flag="--bf16"
  fi
  echo "training AMP choice: $amp_flag"
  python train_task2_v2.py \
    --config "$CONFIG" \
    --device cuda \
    --workers "${V0_DATALOADER_WORKERS:-8}" \
    "$amp_flag" \
    --evaluation-phase validation_screen \
    2>&1 | tee "$RUN_DIR/logs/03_train.log"
  require_file "$RUN_DIR/best_model.pt"
  require_file "$RUN_DIR/run_summary.json"
}

run_importance() {
  require_file "$CONFIG"
  require_file "$RUN_DIR/best_model.pt"
  check_gpu
  mkdir -p "$IMPORTANCE_DIR" "$RUN_DIR/logs"

  local amp_args=()
  if python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    amp_args+=(--bf16)
  fi
  python analyze_modality_importance_v2.py \
    --config "$CONFIG" \
    --checkpoint "$RUN_DIR/best_model.pt" \
    --out-dir "$IMPORTANCE_DIR" \
    --split val \
    --methods ablation \
    --permutation-repeats 1 \
    --bootstrap-samples 500 \
    --seed 42 \
    --batch-size "${V0_EVAL_BATCH_SIZE:-16}" \
    --num-workers "${V0_DATALOADER_WORKERS:-8}" \
    --device cuda \
    "${amp_args[@]}" \
    2>&1 | tee "$RUN_DIR/logs/04_importance.log"
  require_file "$IMPORTANCE_DIR/importance_summary.csv"
  require_file "$IMPORTANCE_DIR/importance_metadata.json"
}

run_summary() {
  require_file "$RUN_DIR/run_summary.json"
  require_file "$IMPORTANCE_DIR/importance_summary.csv"
  require_file "$IMPORTANCE_DIR/importance_metadata.json"
  python summarize_feedback_v0.py \
    --run-dir "$RUN_DIR" \
    --importance-dir "$IMPORTANCE_DIR" \
    --out-dir "$RUN_DIR/feedback_summary" \
    2>&1 | tee "$RUN_DIR/logs/05_summary.log"
}

case "$STAGE" in
  all)
    run_prepare
    run_train
    run_importance
    run_summary
    ;;
  prepare) run_prepare ;;
  train) run_train ;;
  importance) run_importance ;;
  summary) run_summary ;;
esac

echo "V0 stage complete: $STAGE"
echo "results: $RUN_DIR/feedback_summary"

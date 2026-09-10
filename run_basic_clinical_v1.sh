#!/usr/bin/env bash
# Frozen seed-42 teachers; fresh full/basic/no Clinical validation inference.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash run_basic_clinical_v1.sh [check|test|run]
  check  Check every selected task's actual input/checkpoint lineage and mapping
         on CPU; no inference or output creation (default).
  test   Run focused CPU tests; no study data or GPU required.
  run    Run CPU tests, preflight every task, require CUDA/bf16, then recompute
         full/basic/no Clinical predictions from a fresh raw-input pass.

Environment:
  TASKS                  moca_binary cvd_broad_binary cvd_strict_binary
  CODE_ROOT              /home/pzhao3/ot2
  OUT_ROOT               /data/pzhao3/ot2_output
  FULL_ROOT              $OUT_ROOT/model_full_assets_v1
  BASIC_ROOT             $OUT_ROOT/model_full_assets_basic_clinical_v1
  PYTHON_BIN             python
  CONDA_ENV_NAME         ot2 (activated when the existing cluster hook is present)
  BASIC_BATCH_SIZE       2
  BASIC_WORKERS          2
  BASIC_BOOTSTRAPS       2000

Results: BASIC_ROOT/<task>/seed42/val_basic_clinical/
Logs:    BASIC_ROOT/logs/<task>.val_basic_clinical.<unique suffix>
Existing result directories are always refused. Use a fresh BASIC_ROOT to rerun.
Every run builds a NEW native cache inside its NEW result directory. It never
loads previous predictions or caches. Output-root overrides do not rewrite
paths in the YAML configs. All eight non-clinical modalities remain unchanged.
The installed snapshot is hash-verified before any stage.
HELP
}

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" && -r "$1" ]] || die "missing or unreadable file: $1"; }

[[ $# -le 1 ]] || { usage >&2; exit 2; }
STAGE="${1:-check}"
case "$STAGE" in
  -h|--help|help) usage; exit 0 ;;
  check|test|run) ;;
  *) usage >&2; exit 2 ;;
esac

CODE_ROOT="${CODE_ROOT:-/home/pzhao3/ot2}"
OUT_ROOT="${OUT_ROOT:-/data/pzhao3/ot2_output}"
FULL_ROOT="${FULL_ROOT:-$OUT_ROOT/model_full_assets_v1}"
BASIC_ROOT="${BASIC_ROOT:-$OUT_ROOT/model_full_assets_basic_clinical_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS-moca_binary cvd_broad_binary cvd_strict_binary}"
BASIC_BATCH_SIZE="${BASIC_BATCH_SIZE:-2}"
BASIC_WORKERS="${BASIC_WORKERS:-2}"
BASIC_BOOTSTRAPS="${BASIC_BOOTSTRAPS:-2000}"

if [[ -r /home/pzhao3/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/pzhao3/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-ot2}"
fi
command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"
[[ -d "$CODE_ROOT" ]] || die "CODE_ROOT is not a directory: $CODE_ROOT"
cd "$CODE_ROOT"

python3 "$CODE_ROOT/verify_basic_clinical_bundle_v1.py" --root "$CODE_ROOT"

if [[ "$STAGE" == test || "$STAGE" == run ]]; then
  TEST_MODULES=(
    test_clinical_basic_information_v1.py
    test_analyze_basic_clinical_v1.py
    test_clinical_overlap_inference_v1.py
    test_basic_clinical_runner_v1.py
  )
  for test_module in "${TEST_MODULES[@]}"; do
    require_file "$CODE_ROOT/tests/$test_module"
    echo "[test] $test_module"
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m unittest discover \
      -s "$CODE_ROOT/tests" -p "$test_module" -v
  done
  echo "[test] All focused CPU tests passed."
  if [[ "$STAGE" == test ]]; then exit 0; fi
fi

for variable in BASIC_BATCH_SIZE BASIC_BOOTSTRAPS; do
  [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
done
[[ "$BASIC_WORKERS" =~ ^(0|[1-9][0-9]*)$ ]] || die "BASIC_WORKERS must be non-negative"
read -r -a TASK_LIST <<< "$TASKS"
[[ ${#TASK_LIST[@]} -gt 0 ]] || die "TASKS cannot be empty"

config_for_task() {
  case "$1" in
    moca_binary) printf '%s\n' "$CODE_ROOT/configs/task2_v2_moca_full_assets_v1.yaml" ;;
    cvd_broad_binary) printf '%s\n' "$CODE_ROOT/configs/task2_v2_cvd_broad_full_assets_v1.yaml" ;;
    cvd_strict_binary) printf '%s\n' "$CODE_ROOT/configs/task2_v2_cvd_strict_full_assets_v1.yaml" ;;
    *) die "unknown task in TASKS: $1" ;;
  esac
}
checkpoint_for_task() { printf '%s\n' "$FULL_ROOT/models/$1/seed42/best_model.pt"; }
output_for_task() { printf '%s\n' "$BASIC_ROOT/$1/seed42/val_basic_clinical"; }

ENTRYPOINT="$CODE_ROOT/analyze_basic_clinical_v1.py"
require_file "$ENTRYPOINT"
seen_tasks=" "
CONFIG_ARGS=()
for task in "${TASK_LIST[@]}"; do
  config="$(config_for_task "$task")"
  [[ "$seen_tasks" != *" $task "* ]] || die "duplicate task in TASKS: $task"
  seen_tasks+="$task "
  require_file "$config"
  require_file "$(checkpoint_for_task "$task")"
  if [[ "$STAGE" == run ]]; then
    output_dir="$(output_for_task "$task")"
    [[ ! -e "$output_dir" && ! -L "$output_dir" ]] || die "output already exists: $output_dir; choose a fresh BASIC_ROOT"
  fi
  CONFIG_ARGS+=("$task" "$config")
done

# Check the task/seed binding as well as the per-task analyzer's deeper checks.
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" - "${CONFIG_ARGS[@]}" <<'PY'
# basic_clinical-config-preflight
import sys
from pathlib import Path
from train_task2_v2 import load_supervised_config

for task, path in zip(sys.argv[1::2], sys.argv[2::2]):
    _, config = load_supervised_config(Path(path))
    if config.dataset.task != task or config.seed != 42:
        raise ValueError(f"Expected {task}, seed 42 in {path}")
PY

# Every selected task must pass on CPU before the first GPU analysis starts.
for task in "${TASK_LIST[@]}"; do
  echo "[check][$task] Checking actual validation inputs, checkpoint and 23-column mapping."
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$ENTRYPOINT" \
    --config "$(config_for_task "$task")" \
    --checkpoint "$(checkpoint_for_task "$task")" \
    --out-dir "$(output_for_task "$task")" \
    --seed 42 --batch-size "$BASIC_BATCH_SIZE" \
    --num-workers "$BASIC_WORKERS" \
    --bootstrap-samples "$BASIC_BOOTSTRAPS" \
    --device cpu --check-only
done

if [[ "$STAGE" == check ]]; then
  echo "[check] All selected tasks passed; no GPU inference or result files created."
  exit 0
fi

"$PYTHON_BIN" - <<'PY'
# basic_clinical-cuda-preflight
import torch

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
print("torch:", torch.__version__)
print("cuda:", cuda)
print("gpu:", torch.cuda.get_device_name(0) if cuda else "NONE")
print("bf16:", bf16)
if not bf16:
    raise RuntimeError("This runner requires CUDA with bf16 support")
PY

LOG_ROOT="$BASIC_ROOT/logs"
mkdir -p "$LOG_ROOT"
for task in "${TASK_LIST[@]}"; do
  output_dir="$(output_for_task "$task")"
  [[ ! -e "$output_dir" && ! -L "$output_dir" ]] || die "output appeared after preflight: $output_dir"
  ANALYSIS_ARGS=(
    --config "$(config_for_task "$task")"
    --checkpoint "$(checkpoint_for_task "$task")"
    --out-dir "$output_dir"
    --seed 42 --batch-size "$BASIC_BATCH_SIZE"
    --num-workers "$BASIC_WORKERS"
    --bootstrap-samples "$BASIC_BOOTSTRAPS"
    --device cuda --bf16
  )
  log_path="$(mktemp "$LOG_ROOT/$task.val_basic_clinical.XXXXXX")"
  echo "[run][$task] Frozen seed-42 teacher, validation; results: $output_dir"
  echo "[run][$task] Log: $log_path"
  # The analyzer atomically reserves its result directory after validations.
  "$PYTHON_BIN" "$ENTRYPOINT" "${ANALYSIS_ARGS[@]}" \
    2>&1 | tee "$log_path"
  for artifact in basic_clinical_plan.json basic_clinical_conditions.json \
    basic_clinical_metrics.csv basic_clinical_summary.csv \
    basic_clinical_bootstrap_intervals.csv basic_clinical_paired_predictions.csv \
    basic_clinical_cohort_counts.csv basic_clinical_metadata.json source_token_cache_val.pt; do
    require_file "$output_dir/$artifact"
  done
  echo "[run][$task] Complete."
done

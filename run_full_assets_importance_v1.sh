#!/usr/bin/env bash
# Frozen seed-42 full-assets model importance; validation data only.
# No training, preprocessing, universal reconstruction, or test evaluation.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash run_full_assets_importance_v1.sh [check|ablation|cohorts|families|families_joint|permutation]
  check         Validate selected task files, config lineage, imports and CUDA/bf16.
  ablation      Nine single-modality removals (default).
  cohorts       Nine removals with cohort reports in a separate output directory.
  families      Joint retinal-four and wearable-two removals, evaluated separately.
  families_joint  Optional simultaneous retinal-four + wearable-two removal.
  permutation   Ordinary and matched permutations for each of the nine modalities.

Environment:
  TASKS                  Space-separated tasks; default: all three current tasks.
                         moca_binary cvd_broad_binary cvd_strict_binary
  CODE_ROOT              /home/pzhao3/ot2
  OUT_ROOT               /data/pzhao3/ot2_output
  FULL_ROOT              $OUT_ROOT/model_full_assets_v1
  IMPORTANCE_ROOT        $OUT_ROOT/model_full_assets_importance_v1
  PYTHON_BIN             python
  CONDA_ENV_NAME         ot2 (activated if the existing cluster conda hook exists)
  IMPORTANCE_BATCH_SIZE  2
  IMPORTANCE_WORKERS     2
  IMPORTANCE_REPEATS     200
  IMPORTANCE_BOOTSTRAPS  2000
  IMPORTANCE_OVERWRITE   0; set 1 to explicitly rerun an existing stage directory.

Results: IMPORTANCE_ROOT/<task>/seed42/val_<stage>/
The cohorts stage writes val_ablation_cohorts/ and preserves earlier val_ablation/.
The family stages write val_families/ and val_families_joint/, preserving single removals.
Every analysis stage also writes importance_cohort_{summary,bootstrap_intervals,counts}.csv.
Cache:   IMPORTANCE_ROOT/<task>/seed42/source_token_cache_val.pt
All analysis stages share the analyzer's hash-validated cache. A stale cache
is rejected, never replaced; choose a fresh IMPORTANCE_ROOT after input changes.
Changing output roots does not rewrite dataset paths inside the existing YAMLs.

Example: TASKS=moca_binary bash run_full_assets_importance_v1.sh ablation
Follow-up for all three tasks: bash run_full_assets_importance_v1.sh cohorts
Family removal for all three tasks: bash run_full_assets_importance_v1.sh families
HELP
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
STAGE="${1:-ablation}"
case "$STAGE" in
  -h|--help|help) usage; exit 0 ;;
  check|ablation|cohorts|families|families_joint|permutation) ;;
  *) usage >&2; exit 2 ;;
esac

CODE_ROOT="${CODE_ROOT:-/home/pzhao3/ot2}"
OUT_ROOT="${OUT_ROOT:-/data/pzhao3/ot2_output}"
FULL_ROOT="${FULL_ROOT:-$OUT_ROOT/model_full_assets_v1}"
IMPORTANCE_ROOT="${IMPORTANCE_ROOT:-$OUT_ROOT/model_full_assets_importance_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS-moca_binary cvd_broad_binary cvd_strict_binary}"
IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE:-2}"
IMPORTANCE_WORKERS="${IMPORTANCE_WORKERS:-2}"
IMPORTANCE_REPEATS="${IMPORTANCE_REPEATS:-200}"
IMPORTANCE_BOOTSTRAPS="${IMPORTANCE_BOOTSTRAPS:-2000}"
IMPORTANCE_OVERWRITE="${IMPORTANCE_OVERWRITE:-0}"

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -r "$1" && -f "$1" ]] || die "missing or unreadable file: $1"; }
for variable in IMPORTANCE_BATCH_SIZE IMPORTANCE_REPEATS IMPORTANCE_BOOTSTRAPS; do
  [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
done
[[ "$IMPORTANCE_WORKERS" =~ ^(0|[1-9][0-9]*)$ ]] || die "IMPORTANCE_WORKERS must be non-negative"
[[ "$IMPORTANCE_OVERWRITE" == 0 || "$IMPORTANCE_OVERWRITE" == 1 ]] || die "IMPORTANCE_OVERWRITE must be 0 or 1"
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
output_for_task() {
  local output_stage="$STAGE"
  if [[ "$STAGE" == cohorts ]]; then
    output_stage="ablation_cohorts"
  fi
  printf '%s\n' "$IMPORTANCE_ROOT/$1/seed42/val_$output_stage"
}

# Validate every selected task before importing torch or starting any analysis.
require_file "$CODE_ROOT/analyze_modality_importance_v2.py"
seen_tasks=" "
CONFIG_ARGS=()
for task in "${TASK_LIST[@]}"; do
  config="$(config_for_task "$task")"
  [[ "$seen_tasks" != *" $task "* ]] || die "duplicate task in TASKS: $task"
  seen_tasks+="$task "
  require_file "$config"
  require_file "$(checkpoint_for_task "$task")"
  if [[ "$STAGE" != check && "$IMPORTANCE_OVERWRITE" == 0 ]]; then
    output_dir="$(output_for_task "$task")"
    [[ ! -e "$output_dir" && ! -L "$output_dir" ]] || die "output already exists: $output_dir; use a new IMPORTANCE_ROOT or set IMPORTANCE_OVERWRITE=1"
  fi
  CONFIG_ARGS+=("$task" "$config")
done

if [[ -r /home/pzhao3/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/pzhao3/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-ot2}"
fi
command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"
cd "$CODE_ROOT"

# Config loading/lineage checks do not instantiate datasets or iterate any split.
# Exact checkpoint/cohort/cache identities are checked by the analyzer at runtime.
"$PYTHON_BIN" - "${CONFIG_ARGS[@]}" <<'PY'
import sys
from pathlib import Path

import torch
import analyze_modality_importance_v2  # Verify the analysis dependency chain.
from train_task2_v2 import load_supervised_config
from task2_v2_config import validate_model_v2_lineage

for task, config_path in zip(sys.argv[1::2], sys.argv[2::2]):
    _, config = load_supervised_config(Path(config_path))
    validate_model_v2_lineage(config)
    if config.dataset.task != task or config.seed != 42:
        raise ValueError(f"Expected {task}, seed 42 in {config_path}")
    if not config.dataset.master_csv.is_file():
        raise FileNotFoundError(config.dataset.master_csv)
    if not config.dataset.artifact_root.is_dir():
        raise FileNotFoundError(config.dataset.artifact_root)
    print(f"[check][{task}] Config and input lineage ready: {config_path}")

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
print("torch:", torch.__version__)
print("cuda:", cuda)
print("gpu:", torch.cuda.get_device_name(0) if cuda else "NONE")
print("bf16:", bf16)
if not bf16:
    raise RuntimeError("This runner requires CUDA with bf16 support")
PY

if [[ "$STAGE" == check ]]; then
  echo "[check] Ready for validation importance; no data split was iterated."
  exit 0
fi

METHODS=(ablation)
if [[ "$STAGE" == permutation ]]; then
  METHODS=(permutation matched_permutation)
fi
CONDITION_ARGS=(--methods "${METHODS[@]}")
if [[ "$STAGE" == families ]]; then
  CONDITION_ARGS+=(--ablation-groups retinal_all wearable_all)
elif [[ "$STAGE" == families_joint ]]; then
  CONDITION_ARGS+=(--ablation-groups retinal_wearable_all)
fi
for task in "${TASK_LIST[@]}"; do
  output_dir="$(output_for_task "$task")"
  cache_path="$IMPORTANCE_ROOT/$task/seed42/source_token_cache_val.pt"
  mkdir -p "$IMPORTANCE_ROOT/$task/seed42"
  if [[ "$IMPORTANCE_OVERWRITE" == 1 ]]; then
    mkdir -p "$output_dir"
  else
    # Atomic reservation also catches another invocation starting after preflight.
    mkdir "$output_dir"
  fi
  echo "[$STAGE][$task] Frozen seed-42 model, split=val; results: $output_dir"
  "$PYTHON_BIN" "$CODE_ROOT/analyze_modality_importance_v2.py" \
    --config "$(config_for_task "$task")" \
    --checkpoint "$(checkpoint_for_task "$task")" \
    --out-dir "$output_dir" \
    --split val \
    --source-token-cache "$cache_path" \
    "${CONDITION_ARGS[@]}" \
    --permutation-repeats "$IMPORTANCE_REPEATS" \
    --bootstrap-samples "$IMPORTANCE_BOOTSTRAPS" \
    --seed 42 \
    --batch-size "$IMPORTANCE_BATCH_SIZE" \
    --num-workers "$IMPORTANCE_WORKERS" \
    --device cuda \
    --bf16 \
    2>&1 | tee "$output_dir/run.log"
  for artifact in importance_summary.csv importance_repeat_metrics.csv \
    importance_paired_predictions.csv importance_bootstrap_intervals.csv \
    importance_permutation_pairs.csv importance_permutation_audits.json \
    importance_cohort_summary.csv importance_cohort_bootstrap_intervals.csv \
    importance_cohort_counts.csv \
    importance_metadata.json; do
    require_file "$output_dir/$artifact"
  done
  require_file "$cache_path"
  echo "[$STAGE][$task] Complete."
done

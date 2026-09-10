#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
if [[ -r /home/pzhao3/miniforge3/etc/profile.d/conda.sh ]]; then
  source /home/pzhao3/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-ot2}"
fi
export CODE_ROOT
exec "${PYTHON_BIN:-python}" "$CODE_ROOT/run_basic_retrain_v1.py" "$@"

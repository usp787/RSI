#!/bin/bash
# Shared runtime setup for compute-node jobs. This file is sourced by sbatch files.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit jobs from the repository root}"
cd "$PROJECT_ROOT"

: "${CODE_COMMIT:?submit with --export=ALL,CODE_COMMIT=$(git rev-parse HEAD),...}"
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$CODE_COMMIT" ]]; then
  echo "Commit mismatch: requested=$CODE_COMMIT checkout=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Cluster checkout is dirty; refusing to run:" >&2
  git status --short >&2
  exit 2
fi

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/rsi-restem"

export PYTHONNOUSERSITE=1
export RSI_ROOT="${RSI_ROOT:-/scratch/$USER/rsi}"
export HF_HOME="${HF_HOME:-$RSI_ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RSI_ROOT/cache}"
export TMPDIR="${TMPDIR:-$RSI_ROOT/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RSI_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$RSI_ROOT/data" "$RSI_ROOT/artifacts" \
  "$RSI_ROOT/checkpoints" "$RSI_ROOT/environments"

echo "code_commit=$actual_commit"
echo "host=$(hostname) job_id=${SLURM_JOB_ID:-none} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "rsi_root=$RSI_ROOT"

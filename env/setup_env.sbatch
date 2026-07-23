#!/bin/bash
#SBATCH --job-name=rsi-setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
: "${CODE_COMMIT:?submit with --export=ALL,CODE_COMMIT=$(git rev-parse HEAD)}"
actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$CODE_COMMIT" ]] || {
  echo "Commit mismatch: requested=$CODE_COMMIT checkout=$actual_commit" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Cluster checkout is dirty; refusing environment setup" >&2
  git status --short >&2
  exit 2
}

module load cuda/12.8.0
module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
ENV_PATH="$HOME/.conda/envs/rsi-restem"
if [[ ! -x "$ENV_PATH/bin/python" ]]; then
  conda create -y -p "$ENV_PATH" -c conda-forge --override-channels python=3.11 pip
fi
conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1
export RSI_ROOT="${RSI_ROOT:-/scratch/$USER/rsi}"
export HF_HOME="${HF_HOME:-$RSI_ROOT/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$RSI_ROOT/cache"
export TMPDIR="$RSI_ROOT/tmp"
mkdir -p "$RSI_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$RSI_ROOT/environments" "$RSI_ROOT/data" \
  "$RSI_ROOT/artifacts" "$RSI_ROOT/checkpoints"

python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
python -m pip check
python -m pip freeze > "$RSI_ROOT/environments/rsi-restem-${CODE_COMMIT}.freeze.txt"
python src/preflight.py --config configs/restem.yaml --experiment smoke_gsm8k_3b
echo "Environment ready: $ENV_PATH"

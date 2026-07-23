#!/bin/bash
# Submit a serialized one-GPU ReST-EM chain from the Explorer login node.
set -euo pipefail

usage() {
  echo "Usage: bash slurm/submit_chain.sh EXPERIMENT [SHARDS]" >&2
  echo "Example: bash slurm/submit_chain.sh smoke_gsm8k_3b 2" >&2
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
EXPERIMENT=$1
SHARDS=${2:-2}
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "SHARDS must be positive" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"
[[ -z "$(git status --porcelain)" ]] || {
  echo "Refusing to submit from a dirty checkout:" >&2
  git status --short >&2
  exit 2
}
CONFIG=configs/restem.yaml
CODE_COMMIT=$(git rev-parse HEAD)

# Read only scalar experiment metadata; this is control-plane work, not an experiment.
PYTHON_BIN="$HOME/.conda/envs/rsi-restem/bin/python"
[[ -x "$PYTHON_BIN" ]] || {
  echo "Missing rsi-restem environment; run env/setup_env.sbatch first" >&2
  exit 2
}
read -r ROUNDS MODE < <(
  PYTHONNOUSERSITE=1 RSI_ROOT="${RSI_ROOT:-/scratch/$USER/rsi}" \
    "$PYTHON_BIN" - "$CONFIG" "$EXPERIMENT" <<'PY'
import sys
from src.common import load_experiment
cfg = load_experiment(sys.argv[1], sys.argv[2])
print(cfg["rounds"], cfg["mode"])
PY
)

export_vars="ALL,CODE_COMMIT=$CODE_COMMIT,CONFIG=$CONFIG,EXPERIMENT=$EXPERIMENT"
submit() {
  local output
  output=$(sbatch --parsable "$@")
  echo "${output%%;*}"
}

data_job=$(submit --export="$export_vars" slurm/01_prepare_data.sbatch)
echo "data=$data_job"
ready_job=$data_job
score_jobs=()

for ((model_round=0; model_round<=ROUNDS; model_round++)); do
  eval_job=$(submit --dependency="afterok:$ready_job" --array="0-$((SHARDS - 1))%1" \
    --export="$export_vars,PHASE=eval,ROUND=$model_round,SHARD_COUNT=$SHARDS" \
    slurm/10_generate.sbatch)
  score_job=$(submit --dependency="afterok:$eval_job" \
    --export="$export_vars,ROUND=$model_round" slurm/20_score_passk.sbatch)
  score_jobs+=("$score_job")
  echo "eval_m${model_round}=$eval_job score_m${model_round}=$score_job"

  if (( model_round < ROUNDS )); then
    train_generation_job=$(submit --dependency="afterok:$score_job" \
      --array="0-$((SHARDS - 1))%1" \
      --export="$export_vars,PHASE=train,ROUND=$model_round,SHARD_COUNT=$SHARDS" \
      slurm/10_generate.sbatch)
    filter_job=$(submit --dependency="afterok:$train_generation_job" \
      --export="$export_vars,ROUND=$model_round" slurm/11_build_sft.sbatch)
    train_job=$(submit --dependency="afterok:$filter_job" \
      --export="$export_vars,ROUND=$model_round" slurm/12_train_sft.sbatch)
    echo "generate_train_r${model_round}=$train_generation_job filter_r${model_round}=$filter_job train_m$((model_round + 1))=$train_job"
    ready_job=$train_job
  fi
done

round_list=$(seq -s, 0 "$ROUNDS")
dependency=$(IFS=:; echo "${score_jobs[*]}")
report_job=$(submit --dependency="afterok:$dependency" \
  --export="$export_vars,ROUNDS=$round_list" slurm/21_report.sbatch)
echo "report=$report_job"
echo "submitted experiment=$EXPERIMENT mode=$MODE commit=$CODE_COMMIT"

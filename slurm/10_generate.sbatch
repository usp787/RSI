#!/bin/bash
#SBATCH --job-name=rsi-generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/generate_%A_%a.out
#SBATCH --error=logs/generate_%A_%a.err

source slurm/common.sh
: "${EXPERIMENT:?missing EXPERIMENT}"
: "${PHASE:?missing PHASE}"
: "${ROUND:?missing ROUND}"
: "${SHARD_COUNT:?missing SHARD_COUNT}"
: "${SLURM_ARRAY_TASK_ID:?submit this file as a Slurm array}"
python src/generate.py --config "${CONFIG:-configs/restem.yaml}" \
  --experiment "$EXPERIMENT" --phase "$PHASE" --round "$ROUND" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$SHARD_COUNT"

#!/bin/bash
#SBATCH --job-name=rsi-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

source slurm/common.sh
: "${EXPERIMENT:?missing EXPERIMENT}"
: "${ROUND:?missing ROUND}"
python src/train_sft.py --config "${CONFIG:-configs/restem.yaml}" \
  --experiment "$EXPERIMENT" --round "$ROUND"

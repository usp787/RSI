#!/bin/bash
#SBATCH --job-name=rsi-filter
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/filter_%j.out
#SBATCH --error=logs/filter_%j.err

source slurm/common.sh
: "${EXPERIMENT:?missing EXPERIMENT}"
: "${ROUND:?missing ROUND}"
python src/build_sft.py --config "${CONFIG:-configs/restem.yaml}" \
  --experiment "$EXPERIMENT" --round "$ROUND"

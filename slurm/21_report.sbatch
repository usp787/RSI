#!/bin/bash
#SBATCH --job-name=rsi-report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/report_%j.out
#SBATCH --error=logs/report_%j.err

source slurm/common.sh
: "${EXPERIMENT:?missing EXPERIMENT}"
: "${ROUNDS:?missing ROUNDS}"
python src/report.py --config "${CONFIG:-configs/restem.yaml}" \
  --experiment "$EXPERIMENT" --rounds "$ROUNDS"

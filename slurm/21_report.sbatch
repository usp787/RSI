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
: "${FINAL_ROUND:?missing FINAL_ROUND}"
REPORT_NAME="${REPORT_NAME:-report}"
python src/report.py --config "${CONFIG:-configs/restem.yaml}" \
  --experiment "$EXPERIMENT" --final-round "$FINAL_ROUND" \
  --output-name "$REPORT_NAME"

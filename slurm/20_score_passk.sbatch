#!/bin/bash
#SBATCH --job-name=rsi-passk
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/passk_%j.out
#SBATCH --error=logs/passk_%j.err

source slurm/common.sh
: "${EXPERIMENT:?missing EXPERIMENT}"
: "${ROUND:?missing ROUND}"
python src/eval_passk.py --config "${CONFIG:-configs/restem.yaml}" \
  --experiment "$EXPERIMENT" --round "$ROUND"

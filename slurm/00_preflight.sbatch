#!/bin/bash
#SBATCH --job-name=rsi-preflight
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

source slurm/common.sh
python -m unittest discover -s tests -v
python src/preflight.py --config "${CONFIG:-configs/restem.yaml}" \
  --experiment "${EXPERIMENT:-smoke_gsm8k_3b}"

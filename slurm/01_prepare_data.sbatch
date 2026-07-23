#!/bin/bash
#SBATCH --job-name=rsi-data
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err

source slurm/common.sh
: "${EXPERIMENT:?missing EXPERIMENT}"
python src/prepare_data.py --config "${CONFIG:-configs/restem.yaml}" --experiment "$EXPERIMENT"

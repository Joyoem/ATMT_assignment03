#!/usr/bin/env bash
#SBATCH --job-name=atmt-lora-data
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "$HOME/projects/atmt_a3"
source .venv/bin/activate
python -u scripts/prepare_opus.py --train-size 200000 --seed 42

#!/usr/bin/env bash
#SBATCH --job-name=atmt-epoch
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
ROOT="$HOME/projects/atmt_a3"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
: "${METHOD:?Missing METHOD}"
: "${EPOCH_LIMIT:?Missing EPOCH_LIMIT}"
RESUME=()
if [ "$EPOCH_LIMIT" -gt 1 ]; then RESUME=(--resume); fi
python -u scripts/train_full7.py --method "$METHOD" \
    --run-dir "runs/full7_${METHOD}_s42" \
    --max-steps 0 --max-epochs 7 --stop-after-epoch "$EPOCH_LIMIT" \
    --warmup-steps 1000 --warmup-start-factor 0.1 \
    --batch-size 64 --max-tokens 4096 \
    --valid-sentences 512 --valid-batch-size 8 --eval-every 10000 --log-every 100 \
    --seed 42 --lr 0.0003 --max-seq-len 300 "${RESUME[@]}"

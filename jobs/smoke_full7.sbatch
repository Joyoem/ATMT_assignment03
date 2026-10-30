#!/usr/bin/env bash
#SBATCH --job-name=atmt-smoke
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
ROOT="$HOME/projects/atmt_a3"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
: "${METHOD:?Missing METHOD}"
python - <<'PYTHON'
import torch
assert torch.cuda.is_available(), 'GPU unavailable in Slurm'
print('GPU:',torch.cuda.get_device_name(0))
PYTHON
python -u scripts/train_full7.py --method "$METHOD" \
    --run-dir "runs/smoke_full7_${METHOD}_${SLURM_JOB_ID}" \
    --max-steps 100 --max-epochs 7 --stop-after-epoch 7 \
    --warmup-steps 20 --warmup-start-factor 0.1 \
    --batch-size 64 --max-tokens 4096 \
    --valid-sentences 256 --valid-batch-size 8 --eval-every 100 --log-every 20 \
    --seed 42 --lr 0.0003 --max-seq-len 300

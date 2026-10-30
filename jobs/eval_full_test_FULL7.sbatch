#!/usr/bin/env bash
#SBATCH --job-name=atmt-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
ROOT="$HOME/projects/atmt_a3"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
# Execute AFTER training completes. Both generated translations and references
# are saved, plus BLEU, chrF, decoding time and sample outputs.
for method in constant linear; do
  python -u scripts/evaluate_checkpoint.py \
    --checkpoint "runs/full7_${method}_s42/checkpoint_best.pt" \
    --split test --limit 0 --max-len 100 \
    --output-prefix "results/full7_${method}_test_full"
done

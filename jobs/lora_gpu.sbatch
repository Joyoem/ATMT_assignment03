#!/usr/bin/env bash
#SBATCH --job-name=atmt-lora
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "$HOME/projects/atmt_a3"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
: "${PHASE:?Missing PHASE}"
case "$PHASE" in
 smoke) python -u scripts/lora_engine.py train --run-dir runs/lora_smoke --train-limit 256 --valid-limit 64 --stop-after-epoch 1 --batch 16 --max-len 64 --log-every 1 ;;
 train) : "${EPOCH:?Missing EPOCH}"; OPTS=(); if [ "$EPOCH" -gt 1 ];then OPTS+=(--resume);fi;python -u scripts/lora_engine.py train --run-dir runs/lora_sk_en_r8_a16 --stop-after-epoch "$EPOCH" --batch 64 --max-len 96 --log-every 100 "${OPTS[@]}" ;;
 zero_shot) python -u scripts/lora_engine.py eval --variant zero_shot --split test --run-dir runs/lora_sk_en_r8_a16 ;;
 lora_eval) python -u scripts/lora_engine.py eval --variant lora --split test --run-dir runs/lora_sk_en_r8_a16 ;;
 *) echo 'Bad PHASE' >&2;exit 2;;
esac

#!/usr/bin/env bash
#SBATCH --job-name=atmt-required-eval
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
mkdir -p results

complete() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    m = json.loads(p.with_suffix(".metrics.json").read_text())
    h = p.with_suffix(".translations.txt").read_text().splitlines()
    r = p.with_suffix(".references.txt").read_text().splitlines()
    assert m["num_sentences"] == 10000
    assert len(h) == len(r) == 10000
    assert m["split"] == "test"
    assert m["decoding"] == "toolkit_greedy"
    assert m["max_len"] == 100
except Exception:
    sys.exit(1)
PY
}

evaluate() {
  checkpoint="$1"
  prefix="$2"

  if complete "$prefix"; then
    echo "SKIP: already complete $prefix"
  else
    echo "EVALUATING: $checkpoint"
    python -u scripts/evaluate_checkpoint.py \
      --checkpoint "$checkpoint" \
      --split test \
      --limit 0 \
      --max-len 100 \
      --output-prefix "$prefix"
  fi
}

evaluate \
  data/extracted/cz-en/checkpoints/checkpoint_best.pt \
  results/teacher_test_full

evaluate \
  runs/full7_constant_s42/checkpoint_best.pt \
  results/full7_constant_test_full

evaluate \
  runs/full7_linear_s42/checkpoint_best.pt \
  results/full7_linear_test_full

echo "ALL_REQUIRED_EVALUATIONS_COMPLETE"

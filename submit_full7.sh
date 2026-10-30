#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/projects/atmt_a3"
cd "$ROOT"
mkdir -p logs tracking runs results
for method in constant linear; do
  if [ -e "runs/full7_${method}_s42/checkpoint_last.pt" ] || [ -e "runs/full7_${method}_s42/summary.json" ]; then
    echo "ABORT: Existing run for $method. Do not overwrite. Inspect or resume manually." >&2
    exit 1
  fi
done
for method in constant linear; do
  smoke=$(sbatch --parsable --job-name="atmt-smoke-${method}" \
       --export=ALL,METHOD="$method" jobs/smoke_full7.sbatch)
  echo "$method smoke: $smoke" | tee -a tracking/full7_submitted_jobs.txt
  prev="$smoke"
  for epoch_limit in $(seq 1 7); do
    job=$(sbatch --parsable --job-name="atmt-${method}-e${epoch_limit}" \
       --dependency="afterok:$prev" --export=ALL,METHOD="$method",EPOCH_LIMIT="$epoch_limit" \
       jobs/epoch_full7.sbatch)
    echo "$method epoch ${epoch_limit}/7: $job (afterok:$prev)" | tee -a tracking/full7_submitted_jobs.txt
    prev="$job"
  done
done
echo 'Jobs queued. Monitor with: squeue -u "$USER"; see logs/slurm-*.out'

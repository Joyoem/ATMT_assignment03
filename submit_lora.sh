#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/projects/atmt_a3"
mkdir -p logs tracking
# Data preparation can start immediately without taking GPU.
datajob=$(sbatch --parsable jobs/lora_prepare.sbatch)
echo "data preparation job=$datajob"
# After Constant's last epoch (123), the 16GiB RAM reservation disappears.
# The remaining Linear job (10GiB) and LoRA job (10GiB) can then coexist.
# This is intentionally queued: node has 28GB system RAM; currently Constant=16GiB + Linear=10GiB.
prev=$(sbatch --parsable --dependency="afterok:${datajob}:123" --export=ALL,PHASE=smoke jobs/lora_gpu.sbatch)
echo "smoke job=$prev"
for epoch in $(seq 1 5); do
  job=$(sbatch --parsable --dependency="afterok:$prev" --export=ALL,PHASE=train,EPOCH="$epoch" --job-name="atmt-lora-e${epoch}" jobs/lora_gpu.sbatch)
  echo "epoch $epoch job=$job after=$prev";prev=$job
done
for phase in zero_shot lora_eval;do
  job=$(sbatch --parsable --dependency="afterok:$prev" --export=ALL,PHASE="$phase" --job-name="atmt-${phase}" jobs/lora_gpu.sbatch)
  echo "$phase job=$job after=$prev";prev=$job
done
printf 'Data=%s\nFinal=%s\n' "$datajob" "$prev" | tee tracking/lora_submission.txt

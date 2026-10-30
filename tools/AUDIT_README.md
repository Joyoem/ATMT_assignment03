# ATMT Assignment 3 — Detailed audit v2

This is a **read-only audit of original experiment artifacts**, except that it writes new files under `tracking/audit_v2/`. It does not launch experiments, deserialize checkpoints, overwrite logs, or read/write a GitHub repository. It uses the real filenames and schema from the ATMT A3 Warmup and LoRA scripts supplied earlier.

## Run

1. On your Mac, upload `audit_atmt_a3_results_v2.py` to `~/projects/atmt_a3/` on the cluster.
2. On the cluster:

```bash
cd ~/projects/atmt_a3
source .venv/bin/activate
python -u audit_atmt_a3_results_v2.py | tee tracking/audit_v2_terminal.log
```

Optional: `--no-score-recompute` if `sacrebleu` is unavailable. In the intended `.venv`, `sacrebleu` is available.

## Outputs

- `tracking/audit_v2/audit.txt` — human-readable results and PASS/WARN/FAIL/MANUAL flags.
- `tracking/audit_v2/audit.json` — complete machine-readable evidence summary.
- `tracking/audit_v2/comparison.csv` — metrics table for Task 1 and Task 2.
- `tracking/audit_v2/qualitative_candidates.csv` — 8 test sentences improving most and 8 degrading most by **sentence chrF**, for **manual review** (only if all test outputs and sacrebleu exist).
- `tracking/audit_v2/atmt_a3_evidence_v2.zip` — portable evidence: configurations, step/epoch logs, selected code and Slurm scripts, translations, metrics, and optional plots. No `.pt` weights, venv, raw large dataset or full Slurm logs.

**Important:** This is an evidence completeness audit, not a formal correctness proof of the experiment. It will flag missing teacher evaluation, incomplete tests, split inconsistencies and absent report requirements, but it cannot write an honest scientific interpretation or confirm academic provenance automatically.

## Interpreting time

- Task 1 total epoch time is reconstructed by summing `epoch_completed.process_seconds` across **seven separate Slurm jobs**; this is closer to total training/process time than the final `summary.json.process_seconds`. It is still not a full account of scheduling queue time.
- Task 2 sums `train_elapsed_seconds` across five epochs; note that the source code measures this after epoch validation, so this is training **plus** epoch validation time.
- Inference times are read directly from the corresponding test metrics files. Do not compare teacher runtime with newly generated outputs without matching decoding maximum length, hardware and environment.

## Required human review for assignment

The 2–3-page PDF still needs a description of setup, cited guides and data provenance, changed code and hyperparameters, training & inference time impacts, tables/plots, comparison with the baseline, qualitative translation examples, limitations and reflection. Include both full names and alphabetical OLAT usernames, GitHub repo URL, and submit by Nov 3, 2026 23:59.

The student's GitHub repository must contain **source code changes, preprocessing/postprocessing scripts and test translations**, not merely the audit ZIP. Commit your verified version of the scripts to your own repository; never upload the `.pt` files, full OPUS corpus or `.venv`.

# ATMT Assignment 3 — Improving a Baseline NMT System

**Course:** Advanced Techniques of Machine Translation, University of Zurich (Fall 2026)  

This repository contains my code, settings, logs, and test translations for two tasks:

- **Task 1 — Learning-rate warmup:** Compare constant and linear warmup for Czech-to-English translation. Both models were trained from scratch for seven epochs.
- **Task 2 — LoRA:** Adapt the provided Czech-to-English model to Slovak-to-English using low-rank adapters. Compare the original model (zero-shot) with the adapted model.

## Main results

| Task | Model | Test sentences | BLEU ↑ | chrF ↑ |
|---|---|---:|---:|---:|
| 1 | Original course baseline | 10,000 | 20.14 | 44.46 |
| 1 | Constant warmup | 10,000 | **22.58** | **44.85** |
| 1 | Linear warmup | 10,000 | 21.72 | 43.60 |
| 2 | Zero-shot baseline | 2,000 | 8.06 | 29.49 |
| 2 | LoRA adapted | 2,000 | **14.29** | **34.44** |

Task 1 uses matched test data and greedy decoding. Constant and linear runs use the same initialization, optimizer, data order, and training budget. The original baseline is a reference, but its training setup was not identical. Task 2 uses the same test sentences and decoding settings for both models.

## Repository guide

| Location | What it contains |
|---|---|
| `scripts/train_full7.py` | Task 1 training and the two learning-rate schedules |
| `scripts/evaluate_checkpoint.py` | Task 1 test translation and BLEU/chrF evaluation |
| `scripts/prepare_opus.py` | Download and prepare OPUS-100 Slovak–English data |
| `scripts/lora_engine.py` | LoRA layers, fine-tuning, and evaluation |
| `scripts/preflight_lora_cpu.py` | Small CPU check for the LoRA implementation |
| `scripts/summarize_full7.py` | Task 1 summary/plot helper (use audited totals for seven-epoch time) |
| `jobs/` | Slurm scripts for training, checks, and evaluation |
| `submit_full7.sh`, `submit_lora.sh` | Original cluster job submission scripts |
| `runs/full7_*/` | Task 1 settings, learning-rate/training logs, and validation records |
| `runs/lora_sk_en_r8_a16/` | Task 2 settings, epoch logs, test scores, and translations |
| `results/` | Task 1 full-test metrics and 10,000-line translations for each model |
| `tracking/audit_v2/` | Final audit, result table, and selected translation examples |
| `tracking/experiment_plan_full7.md` | Experimental protocol and important limits |
| `evidence/` | OPUS-100 data manifest, software versions, and short Slurm log excerpts |
| `tools/` | Audit and safe repository-sync utilities |
| `report/main.tex` | LaTeX source for the 2–3 page report (edit names before submission) |

**Quick results:** See `tracking/audit_v2/comparison.csv` for the score table and `tracking/audit_v2/audit.txt` for the completeness check (47 OK, 2 WARN, 0 FAIL). For LoRA, see `runs/lora_sk_en_r8_a16/*test_metrics.json`.

## How to use the code

1. Get the original [course toolkit](https://github.com/YinZhengxun/atmt_2025). The recorded toolkit commit is in `tracking/toolkit_commit.txt`.
2. Obtain the original course Czech–English data, SentencePiece tokenizers, and baseline checkpoint through the course. Obtain OPUS-100 `en-sk` for Task 2 (the source URL and checksums are in `evidence/opus100_en_sk_manifest.json`).
3. Install the Python packages used by the experiments: PyTorch, NumPy, SentencePiece, SacreBLEU, and tqdm. See `evidence/runtime_versions.json` for the recorded versions.
4. Put the data and toolkit in the paths expected by the scripts, or change the paths for your machine. The Slurm scripts use `~/projects/atmt_a3` and a `.venv` environment. **They are cluster-specific.**

Useful commands in the prepared cluster environment:

```bash
# Prepare Slovak–English data (Task 2)
python scripts/prepare_opus.py --train-size 200000 --seed 42

# Evaluate the already trained Task 1 checkpoints
sbatch jobs/eval_required_a3.sbatch

# Recheck the experiment records
python tools/audit_atmt_a3_results_v2.py --root ~/projects/atmt_a3
```

`submit_full7.sh` was used to schedule the seven-epoch runs. `submit_lora.sh` contains an **old Slurm job-ID dependency** and should be edited before reuse; do not run it unchanged on another cluster. Training requires the original checkpoint/data and GPU resources. The repository does not contain model weights.

## Data, limitations, and submission

- Task 1: 2,000,000 Czech–English training pairs; seven epochs; seed 42; 512 validation sentences; 10,000 test sentences.
- Task 2: OPUS-100 Slovak–English; 200,000 training pairs; 2,000 validation and 2,000 test sentences; five LoRA epochs.
- We ran one seed per method. These results do not show statistical significance.
- The original OPUS-100 validation and test splits contain **10 identical sentence pairs**. We kept the official splits and recorded the overlap.



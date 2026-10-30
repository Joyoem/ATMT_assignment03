# ATMT Assignment 3 — Improving a Baseline NMT System

Course: Advanced Techniques of Machine Translation, University of Zurich, Fall 2026.

**Task 1: Learning-rate warmup.** Compare constant and linear warmup under matched Czech→English Transformer initialization, optimizer, training data and step budget (seven epochs). Record learning-rate and training-loss trajectories, validation negative log-likelihood/perplexity, checkpoints, training cost and final test translations/quality when available.

**Task 2: LoRA adaptation.** Adapt the supplied Czech→English Transformer to Slovak→English using OPUS-100 `en-sk` training data, original Czech/English SentencePiece vocabularies and low-rank adapters in attention projections. Compare an unmodified teacher checkpoint (zero-shot) with its LoRA-adapted counterpart on the **same untouched test split**, reporting BLEU, chrF, runtime and trainable parameters.

## Where to look

- `scripts/` and `jobs/` — actual modified implementations and Slurm launch scripts, copied from the experiment directory.
- `runs/` — configuration and training/validation records, excluding heavy weights.
- `results/` — Task 1 translated test outputs and matched metrics, when evaluation has completed.
- `tracking/audit_v2/` — evidence completeness audit (`audit.txt`), machine-readable audit (`audit.json`), comparison table and qualitative-analysis candidates.
- `evidence/` — data provenance, manifests, selected Slurm log excerpts and hashes for full local logs.
- `tools/` — audit and safe synchronization scripts for refreshing this repository from the GPU cluster.

**Status:** The presence of an output file does not alone establish experimental correctness. Check `tracking/audit_v2/audit.txt` for PASS/WARN/FAIL and review all results before citing them in the report.

## Reproducing

The course's base source is [YinZhengxun/atmt_2025](https://github.com/YinZhengxun/atmt_2025), with an exact commit recorded in `tracking/toolkit_commit.txt` when archived. Download the baseline checkpoint and original Czech–English tokenizers from the course's sanctioned distribution, and obtain OPUS-100 `en-sk` separately. Place data at the paths expected by the job scripts or update paths for your environment. Do **not** train new source vocabularies when loading the teacher checkpoint.

Original checkpoints (`.pt`), private/course data, the full OPUS archive, Conda/venv contents and full verbose Slurm logs are kept **outside GitHub**. The dataset manifests and artifact hashes support traceability without redistributing these materials.

Assignment deadline: November 3, 2026, 23:59. The report (2–3 pages, including figures/tables) is submitted separately in OLAT with both contributors' names/usernames and this repository URL.

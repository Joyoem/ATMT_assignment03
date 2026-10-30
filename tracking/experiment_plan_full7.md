# ATMT Assignment 3 — Experiment protocol (2026-10-29)

## Research question
Does the *shape* of the LR warmup influence early convergence and translation quality in a Czech-English Transformer NMT system?

## Independent variable
`method=constant` vs `method=linear`, keeping the final learning rate 0.0003 and initial LR factor 0.1 identical.
- Constant: LR=0.00003 during training steps 1–1000; LR=0.0003 thereafter.
- Linear: linearly interpolate 0.00003 at step 1 to 0.0003 at step 1000; LR=0.0003 thereafter.

Both use Adam and a fixed schedule after the warmup. These are explicit *operational definitions*, not a claim that every paper defines "constant warmup" this way.

## Controlled variables
- Source and target: same 2,000,000 pre-tokenized aligned Czech/English training pairs; same SentencePiece vocabularies.
- Model: 3-layer encoder / 3-layer decoder Transformer, 256 hidden size, 4 heads, 1024 FFN, 0.1 dropout, max length 300; freshly randomly initialized in both runs. No teacher checkpoint weights are used for Task 1.
- Seed 42 (both initialization and sampler), Adam optimizer, lr 3e-4, clip norm 4, batch size <=64, max batch tokens 4096, 7 complete epochs.
- Seeded *uniform* corpus permutation with local length bucketing: this is an intentional deviation from the original toolkit's global length-sort BatchSampler, which could bias an incomplete epoch toward short sequences. Both treatments use exactly the same ordering.
- Validation: 512 evenly-spaced dev sentences with teacher forcing every 10,000 steps and at epoch end, normalized negative log likelihood per non-pad token and perplexity.
- All parameter changes vs toolkit are logged in `config.json`, along with toolkit git commit and GPU / library versions.

## Full-epoch protocol
Each of the seven epochs processes the entire 2,000,000-pair training split once. Actual optimizer steps vary with maximum-token bucketing and are logged; per-run sample appearances should total 14,000,000. This is a more comparable dataset-pass budget than the 12,000-step pilot, but the batching/sampling and seed still differ from the teacher baseline.

## Outputs per run
- `config.json`: exact effective hyperparameters, toolkit commit, environment and tokenizer SHA256.
- `validation_indices.json`: subset IDs.
- `train_steps.csv`: global step, LR, token-normalized train NLL, original-toolkit-scaled train loss, grad norm, batch sizes, tokens, timing and peak allocated VRAM (sampled at log events).
- `events.jsonl`: dev NLL / perplexity and evaluation timing on every checkpoint step.
- `checkpoint_last.pt` and `checkpoint_best.pt`: actual model and optimizer weights for reproducibility; periodic last save overwrites to bound disk usage.
- `summary.json`: runtime and best dev loss.
- `results/*.metrics.json`, `*.translations.txt`, `*.references.txt`: fixed-subset greedy validation and later full test translations.

## Hypotheses and interpretation
- H1: a gradual ramp may reduce training instability compared with a sharp jump at warmup end; inspect train loss, grad norm, valid loss, and LR around step 1000.
- H2: lower dev loss need not imply better BLEU. Compute validation BLEU after training and full test BLEU only for selected final checkpoints.
- Two single-seed runs support exploratory conclusions only, not statistical significance. If time allows, repeat with seed 123.
- Other explanation: constant strategy accumulates less LR during warmup, so cumulative step size differs; discuss as a confound inherent to the chosen comparison.

## Task 2 (separate work)
Current `seq2seq/decode.py` is greedy (`argmax`). The `seq2seq/beam.py` provided by the 2025 toolkit is based on RNN/LSTM state and is NOT integrated into the Transformer decoder. Implement beam search separately and compare using the exact same teacher checkpoint and validation/test samples. Do not claim `beam_size=5` from the 2026 teacher's log proves beam search was used.

## GitHub / submission
Keep data archives, `.pt` weights, virtualenvs and SLURM huge logs out of GitHub. Include original code attribution, modifications, scripts, experimental configs, figures, and the assignment-required test set translation files. Commit all before November 3 23:59 Europe/Zurich.

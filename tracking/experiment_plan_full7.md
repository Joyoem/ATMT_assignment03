# ATMT Assignment 3 — Experimental protocol

This file describes the experiments that were **actually completed**. It replaces an earlier draft that mentioned beam search. We did not implement beam search.

## Task 1 — Learning-rate warmup (Czech → English)

**Question:** Does the warmup schedule affect convergence and final translation quality?

- Models: two Transformers trained **from scratch**, with the same initial weights (seed 42), training order, tokenizers, Adam optimizer, and 7-epoch budget.
- Training data: 2,000,000 aligned Czech–English sentence pairs; all pairs processed once per epoch.
- Model: 3 encoder and 3 decoder layers, hidden size 256, 4 attention heads, FFN size 1024, dropout 0.1.
- Learning rate after warmup: 0.0003; warmup length: 1,000 steps.
- **Constant:** learning rate 0.00003 for steps 1–1000; then 0.0003.
- **Linear:** increase from 0.00003 at step 1 to 0.0003 at step 1000; then 0.0003.
- Other settings: batch size up to 64; at most 4,096 tokens per batch; gradient clipping 4; max sequence length 300; 7 full epochs (226,854 optimizer steps).
- Validation: same fixed 512-sentence subset, teacher-forced NLL and perplexity. Test: same 10,000 sentences, greedy decoding, max length 100, BLEU and chrF.
- Original course checkpoint evaluated on the same test set as a **reference**, not a fully matched training control.

**Results:** Constant final validation NLL 2.0467, test BLEU 22.5844, chrF 44.8521. Linear final validation NLL 2.0722, test BLEU 21.7178, chrF 43.6010. Linear was slightly better early, but constant was better at the end. Training time was about 7,414 and 7,435 seconds, respectively (sum of seven epoch records).

**Limit:** One random seed per schedule. The schedules also differ in cumulative learning rate during warmup. We cannot claim statistical significance or a proven causal mechanism beyond the controlled schedule comparison.

## Task 2 — LoRA adaptation (Slovak → English)

**Question:** Can a small set of LoRA parameters improve Slovak-to-English translation from the Czech-to-English baseline?

- Original course Czech–English checkpoint is the teacher; it is kept frozen during LoRA training.
- Reuse the original Czech source and English target SentencePiece tokenizers. No new tokenizer is trained.
- Data: official OPUS-100 `en-sk` splits. Shuffle the training split with seed 42 and take 200,000 pairs; use 2,000 dev and 2,000 test pairs.
- Insert rank-8, alpha-16 LoRA updates into query/value projections in nine attention modules (18 adapted linear layers).
- Train 73,728 of 11,905,600 parameters (0.619%). AdamW, learning rate 0.0003, batch size 64, five epochs.
- Evaluate the unmodified teacher (zero-shot) and adapted model on the same 2,000 test sentences, with greedy decoding and max generation length 95.

**Results:** Zero-shot BLEU 8.0632 and chrF 29.4903. LoRA BLEU 14.2926 and chrF 34.4390. Training and validation took about 836.7 seconds in total.

**Limit:** The OPUS dev/test sets have 10 identical sentence pairs. Some adapted translations still lose names or change meaning. The inference time difference should not be treated as a model-speed gain because output lengths differ.

## Files and audit

- Code: `scripts/`, `jobs/`, `submit_full7.sh`, `submit_lora.sh`.
- Training and validation: `runs/`.
- Test results and translations: `results/` (Task 1), `runs/lora_sk_en_r8_a16/` (Task 2).
- Final scores and verification: `tracking/audit_v2/`.
- Data and runtime records: `evidence/`.

The final audit reports **47 OK, 2 WARN, 0 FAIL**. The two warnings are the OPUS dev/test overlap and the uncommitted local course-toolkit changes. We did not run extra seeds, beam search, or full fine-tuning because these were not required for the selected subtasks.

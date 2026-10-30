#!/usr/bin/env python3
"""ATMT A3 evidence audit v2: read-only with respect to all original runs.

Run from the project root (~ /projects/atmt_a3), or give --root.
Outputs ONLY to tracking/audit_v2/. Checks Task 1 / Task 2 completion,
logging consistency, fair comparisons, required submission evidence and
translation alignment. No GPU, checkpoint unpickling or large-corpus copy.

Requires Python stdlib; optional sacrebleu for score verification & examples.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile


def read_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return {'_parse_error': str(e)}


def read_jsonl(path):
    if not path.is_file():
        return []
    out = []
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    out.append({'_parse_error_at_line': i})
    return out


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def fint(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def csv_scan(path, mode):
    result = {'exists': path.is_file(), 'rows': 0, 'first': None, 'last': None,
              'columns': [], 'bad_values': 0, 'nonmonotone_steps': 0,
              'train_nll_summary': {}, 'lr_samples': {}, 'gpu_peak_mb': None,
              'grad_clipped_logged_fraction': None, 'logged_step_seconds_mean': None}
    if not path.is_file():
        return result
    values = {'early_1_1000': [], 'after_1000_10000': [], 'late_10001_plus': []}
    clip_count = 0
    sec_sum = 0.0
    sec_count = 0
    grad_count = 0
    previous = None
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        result['columns'] = reader.fieldnames or []
        for row in reader:
            result['rows'] += 1
            if result['first'] is None:
                result['first'] = row
            result['last'] = row
            step = fint(row.get('step'))
            if step is None or (previous is not None and step <= previous):
                result['nonmonotone_steps'] += 1
            if step is not None:
                previous = step
            nll = fnum(row.get('train_nll_per_token'))
            if nll is None:
                result['bad_values'] += 1
            elif step is not None:
                label = 'early_1_1000' if step <= 1000 else ('after_1000_10000' if step <= 10000 else 'late_10001_plus')
                values[label].append(nll)
            lr = fnum(row.get('lr'))
            if step is not None and lr is not None and (step in (1, 100, 500, 1000, 1100, 10000) or (step <= 1000 and step % 100 == 0)):
                result['lr_samples'][str(step)] = lr
            gpu = fnum(row.get('gpu_peak_mb'))
            if gpu is None:
                gpu = fnum(row.get('peak_gpu_MB'))
            if gpu is not None:
                result['gpu_peak_mb'] = max(result['gpu_peak_mb'] or 0, gpu)
            sec = fnum(row.get('step_seconds'))
            if sec is not None:
                sec_sum += sec
                sec_count += 1
            grad = fnum(row.get('grad_norm'))
            if grad is not None:
                grad_count += 1
                # warmup uses clip_norm=4.0; this counts records where norm >= 4.
                if grad >= 4.0:
                    clip_count += 1
    for name, v in values.items():
        if v:
            result['train_nll_summary'][name] = {'n_logged': len(v), 'mean': statistics.fmean(v),
                                                  'sd': statistics.stdev(v) if len(v) > 1 else None,
                                                  'min': min(v), 'max': max(v)}
    if sec_count:
        result['logged_step_seconds_mean'] = sec_sum / sec_count
    if grad_count:
        result['grad_clipped_logged_fraction'] = clip_count / grad_count
    return result


def lines_read(path):
    if not path.is_file():
        return None
    return path.read_text(encoding='utf-8').splitlines()


def line_count(path):
    if not path.is_file():
        return None
    with path.open('rb') as f:
        return sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1024 * 1024), b''))


def file_sha(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_info(path):
    if not (path / '.git').exists():
        return {'is_git_repository': False}
    def run(*args):
        p = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else f'error: {p.stderr.strip()}'
    return {'is_git_repository': True, 'commit': run('rev-parse', 'HEAD'),
            'status_porcelain': run('status', '--porcelain')[:8000],
            'remote': run('remote', '-v')[:800]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--no-score-recompute', action='store_true', help='Skip checking BLEU/chrF with sacrebleu')
    args = p.parse_args()
    root = args.root.resolve()
    destination = root / 'tracking' / 'audit_v2'
    destination.mkdir(parents=True, exist_ok=True)
    messages = []
    checks = []
    details = {'task1': {}, 'task2': {}, 'toolkit_git': git_info(root / 'toolkit')}
    files_for_bundle = set()
    
    def add(text=''):
        messages.append(str(text))

    def check(level, message):
        checks.append({'status': level, 'message': message})
        add(f'[{level}] {message}')

    def include(relative):
        f = root / relative
        if f.is_file():
            files_for_bundle.add(f)
            return True
        return False

    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None
        if not args.no_score_recompute:
            check('WARN', 'sacrebleu not installed: scores cannot be independently recomputed')

    add('ATMT Assignment 3 — Detailed Results and Submission Audit (v2)')
    add(f'Root: {root}')
    add('Only reads original run files; writes to tracking/audit_v2. Absence of Slurm queue entries is NOT a completion proof.')
    add('')

    teacher_ckpt = root / 'data/extracted/cz-en/checkpoints/checkpoint_best.pt'
    check('OK' if teacher_ckpt.is_file() else 'FAIL', 'Original teacher checkpoint is present' if teacher_ckpt.is_file() else 'Original teacher checkpoint missing')

    # TASK 1
    add('\n==== TASK 1: LEARNING RATE WARMUP ====')
    warm = {}
    for method in ('constant', 'linear'):
        run = root / 'runs' / f'full7_{method}_s42'
        cfg = read_json(run / 'config.json') or {}
        summary = read_json(run / 'summary.json') or {}
        events = read_jsonl(run / 'events.jsonl')
        epochs = sorted({fint(e.get('completed_epochs')) for e in events if e.get('type') == 'epoch_completed' and fint(e.get('completed_epochs')) is not None})
        v = [e for e in events if e.get('type') == 'validation' and fnum(e.get('valid_nll_per_token')) is not None]
        best = min(v, key=lambda e: float(e['valid_nll_per_token'])) if v else None
        scans = csv_scan(run / 'train_steps.csv', 'warmup')
        # The per-job 'process_seconds' is monotonic WITHIN each Slurm process only.
        # Each epoch is independently submitted; summing epoch_completed events is a closer wall-clock estimate.
        per_epoch_process_seconds = [fnum(e.get('process_seconds')) for e in events if e.get('type') == 'epoch_completed']
        cumulative_runtime = sum(x for x in per_epoch_process_seconds if x is not None)
        metrics = read_json(root / 'results' / f'full7_{method}_test_full.metrics.json') or {}
        hyp = root / 'results' / f'full7_{method}_test_full.translations.txt'
        ref = root / 'results' / f'full7_{method}_test_full.references.txt'
        h_count, r_count = line_count(hyp), line_count(ref)
        test_count = fint(metrics.get('num_sentences'))
        done = (epochs == list(range(1, 8)) and fint(summary.get('completed_epochs')) == 7 and summary.get('type') == 'completed')
        test_ok = (test_count == 10000 and h_count == r_count == 10000 and metrics.get('split') == 'test' and metrics.get('decoding') == 'toolkit_greedy')
        if done:
            check('OK', f'{method}: 7 epoch-completion records + final completed summary')
        else:
            check('FAIL', f'{method}: missing or inconsistent full 7-epoch completion; epochs seen={epochs}; summary epochs={summary.get("completed_epochs")}')
        seen = fint(summary.get('examples_seen_total'))
        if seen is not None:
            check('OK' if seen == 14_000_000 else 'WARN', f'{method}: total examples seen={seen:,} (expected 14,000,000 for seven full 2M-pair epochs)')
        else:
            check('WARN', f'{method}: no total sentence appearances recorded')
        if scans['bad_values'] or scans['nonmonotone_steps']:
            check('WARN', f'{method}: CSV problems: bad NLL={scans["bad_values"]}, nonmonotone steps={scans["nonmonotone_steps"]}')
        if not scans['rows']:
            check('FAIL', f'{method}: no step-wise training log')
        if not v:
            check('FAIL', f'{method}: no finite validation NLL records')
        if test_ok:
            check('OK', f'{method}: matched full 10,000-sentence test metrics, hypotheses and references')
        else:
            check('FAIL', f'{method}: full 10,000-sentence test incomplete/inconsistent (metric n={test_count}, hypotheses={h_count}, refs={r_count})')
        out = {'config': cfg, 'summary': summary, 'epochs_seen': epochs, 'training_completed': done,
               'validation_records': len(v), 'best_validation': best, 'last_validation': v[-1] if v else None,
               'first_validation': v[0] if v else None, 'csv_summary': scans,
               'epoch_seconds': per_epoch_process_seconds, 'total_epoch_process_seconds': cumulative_runtime,
               'test_metrics': metrics, 'test_hypotheses_lines': h_count, 'test_references_lines': r_count,
               'full_test_verified': test_ok}
        warm[method] = out
        add(f'{method}: epochs={epochs}; final step={summary.get("step")}; train seconds (sum of job-level epoch records)={cumulative_runtime:.1f}; best dev NLL={best.get("valid_nll_per_token") if best else None}; test BLEU={metrics.get("bleu")}; chrF={metrics.get("chrf")}; inference seconds={metrics.get("seconds")}')
        for name in ('config.json', 'summary.json', 'events.jsonl', 'train_steps.csv', 'first_batches.jsonl', 'validation_indices.json'):
            include(f'runs/full7_{method}_s42/{name}')
        for name in ('metrics.json', 'translations.txt', 'references.txt'):
            include(f'results/full7_{method}_test_full.{name}')
        for k in ('checkpoint_best.pt', 'checkpoint_last.pt'):
            if not (run / k).is_file():
                check('WARN', f'{method}: missing {k}')

    a, b = warm['constant'], warm['linear']
    ca, cb = a['config'], b['config']
    init_equal = bool(ca.get('initial_weights_sha256') and ca.get('initial_weights_sha256') == cb.get('initial_weights_sha256'))
    check('OK' if init_equal else 'FAIL', 'Task 1 identical initial model weight hash' if init_equal else 'Task 1 initial model weight hashes differ or missing')
    for key in ('seed', 'lr', 'warmup_steps', 'warmup_start_factor', 'batch_size', 'max_tokens', 'max_epochs', 'sampler', 'source_tokenizer_sha256', 'target_tokenizer_sha256', 'clip_norm', 'max_seq_len'):
        if ca.get(key) != cb.get(key):
            check('FAIL', f'Task 1 controlled variable differs: {key} constant={ca.get(key)!r}, linear={cb.get(key)!r}')
    vals1, vals2 = root / 'runs/full7_constant_s42/validation_indices.json', root / 'runs/full7_linear_s42/validation_indices.json'
    if vals1.is_file() and vals2.is_file():
        check('OK' if file_sha(vals1) == file_sha(vals2) else 'FAIL', 'Task 1 same validation sentence indices')
    else:
        check('WARN', 'Cannot establish that Task 1 validation sentence subsets were identical')
    for step in (1, 100, 500, 1000):
        x, y = a['csv_summary']['lr_samples'].get(str(step)), b['csv_summary']['lr_samples'].get(str(step))
        if x is not None and y is not None:
            add(f'Learning rate @ step {step}: constant={x:.9g} linear={y:.9g}')
    if all(fnum(x['test_metrics'].get('bleu')) is not None for x in (a, b)):
        add(f'Full test BLEU linear - constant: {float(b["test_metrics"]["bleu"]) - float(a["test_metrics"]["bleu"]):+.3f}')
    details['task1'] = warm
    details['task1']['initial_weights_match'] = init_equal
    teacher_metrics = read_json(root / 'results/teacher_test_full.metrics.json')
    if teacher_metrics:
        details['teacher_test_metrics'] = teacher_metrics
        include('results/teacher_test_full.metrics.json')
        include('results/teacher_test_full.translations.txt')
        include('results/teacher_test_full.references.txt')
        check('OK', 'Teacher baseline has a separately evaluated full-test metric file (review decoding settings before comparing)')
    else:
        check('WARN', 'No matched teacher Czech→English baseline evaluation (results/teacher_test_full.metrics.json). Original teacher translations use different max length / runtime; do not claim strictly matched BLEU/time.')

    # TASK 2
    add('\n==== TASK 2: SLOVAK→ENGLISH LoRA ====')
    data = root / 'data/opus100_en_sk'
    manifest = read_json(data / 'manifest.json') or {}
    lrun = root / 'runs/lora_sk_en_r8_a16'
    cfg = read_json(lrun / 'config.json') or {}
    epoch_events = read_jsonl(lrun / 'epochs.jsonl')
    epoch_ids = sorted({fint(e.get('epoch_completed')) for e in epoch_events if fint(e.get('epoch_completed')) is not None})
    epoch_times = [fnum(e.get('train_elapsed_seconds')) for e in epoch_events]
    total_lora_epoch_time = sum(x for x in epoch_times if x is not None)
    ls = csv_scan(lrun / 'steps.csv', 'lora')
    lora_done = epoch_ids == list(range(1, 6))
    check('OK' if lora_done else 'FAIL', f'LoRA 5 epochs completed: {lora_done}; epoch records={epoch_ids}')
    check('OK' if (cfg.get('modules_wrapped') == 18 and fint(cfg.get('trainable_parameters')) == 73728) else 'WARN',
          f'LoRA module/parameter audit: modules={cfg.get("modules_wrapped")}, trainable={cfg.get("trainable_parameters")}, total={cfg.get("total_parameters")}')
    check('OK' if cfg.get('teacher_epoch') is not None else 'WARN', 'LoRA configuration records original teacher checkpoint epoch')
    for cfg_key, expected_value in [('train_sentences', 200000), ('valid_sentences', 2000),
                                   ('rank', 8), ('alpha', 16), ('batch', 64), ('seed', 42)]:
        observed = cfg.get(cfg_key)
        check('OK' if fnum(observed) == expected_value else 'WARN',
              f'LoRA recorded {cfg_key}={observed!r}; planned value={expected_value} (disclose intentional deviations)')
    check('OK' if (lrun / 'checkpoint_best.pt').is_file() else 'FAIL',
          'LoRA best checkpoint saved')
    for split, expected in [('train', 200000), ('dev', 2000), ('test', 2000)]:
        counts = manifest.get('counts', {}).get(split, {})
        source, target = data / f'{split}.sk', data / f'{split}.en'
        nc, nt = line_count(source), line_count(target)
        manifest_n = fint(counts.get('selected_pairs'))
        good = nc == nt == manifest_n and manifest_n is not None
        check('OK' if good else 'FAIL', f'OPUS {split}: Slovak lines={nc}, English lines={nt}, manifest selected={manifest_n}')
        if manifest_n is not None and manifest_n != expected:
            check('WARN', f'OPUS {split}: selected {manifest_n}, expected {expected} for planned experiment; disclose if intentionally changed')
        if source.is_file() and target.is_file() and manifest.get('files_sha256'):
            for lang, f in [('sk', source), ('en', target)]:
                expected_sha = manifest['files_sha256'].get(f'{split}.{lang}')
                if expected_sha:
                    check('OK' if file_sha(f) == expected_sha else 'FAIL', f'OPUS {split}.{lang}: SHA256 matches manifest')
    overlaps = manifest.get('pair_overlap') or {}
    if overlaps:
        leaked = {k: v for k, v in overlaps.items() if fint(v) not in (None, 0)}
        check('WARN' if leaked else 'OK', f'OPUS exact-pair overlap between splits: {overlaps}')
    else:
        check('WARN', 'OPUS manifest does not contain train/dev/test pair overlap audit')
    if not ls['rows']:
        check('FAIL', 'No LoRA logged training steps')
    metrics = {}
    for variant in ('zero_shot', 'lora'):
        m = read_json(lrun / f'{variant}_test_metrics.json') or {}
        hypotheses = lrun / f'{variant}_test_translations.en'
        nlines = line_count(hypotheses)
        complete = (m.get('split') == 'test' and m.get('variant') == variant and fint(m.get('n')) == 2000 and nlines == 2000)
        check('OK' if complete else 'FAIL', f'{variant}: 2,000 test translations + matching metrics (metrics n={m.get("n")}, lines={nlines})')
        metrics[variant] = {**m, 'translation_lines': nlines, 'complete': complete}
        include(f'runs/lora_sk_en_r8_a16/{variant}_test_metrics.json')
        include(f'runs/lora_sk_en_r8_a16/{variant}_test_translations.en')
    dec1, dec2 = metrics['zero_shot'].get('decoding'), metrics['lora'].get('decoding')
    if dec1 and dec2:
        check('OK' if dec1 == dec2 else 'FAIL', f'LoRA fair decode settings: zero-shot={dec1!r}; adapted={dec2!r}')
    else:
        check('WARN', 'LoRA decoding settings not recorded in both evaluation files')
    add(f'LoRA: total train+validation epoch seconds (sum of epoch records)={total_lora_epoch_time:.1f}; zero-shot BLEU={metrics["zero_shot"].get("BLEU")}; adapted BLEU={metrics["lora"].get("BLEU")}')
    if all(fnum(metrics[k].get('seconds')) is not None for k in ('zero_shot', 'lora')):
        time_delta = float(metrics['lora']['seconds']) - float(metrics['zero_shot']['seconds'])
        add(f'LoRA inference time delta (adapted - zero-shot): {time_delta:+.2f} seconds')
    if all(fnum(metrics[k].get('BLEU')) is not None for k in ('zero_shot', 'lora')):
        bleu_delta = float(metrics['lora']['BLEU']) - float(metrics['zero_shot']['BLEU'])
        add(f'LoRA BLEU delta (adapted - zero-shot): {bleu_delta:+.3f}')
    details['task2'] = {'manifest': manifest, 'config': cfg, 'epoch_records': epoch_events,
                        'epochs_seen': epoch_ids, 'completed_5_epochs': lora_done,
                        'epoch_seconds': epoch_times, 'total_train_and_validation_seconds': total_lora_epoch_time,
                        'csv_summary': ls, 'test_metrics': metrics}
    for name in ('config.json', 'epochs.jsonl', 'steps.csv'):
        include(f'runs/lora_sk_en_r8_a16/{name}')
    include('data/opus100_en_sk/manifest.json')
    include('tracking/lora_cpu_preflight.log')
    if not (lrun / 'checkpoint_best.pt').is_file():
        check('WARN', 'LoRA checkpoint_best.pt missing')

    # Metrics recomputation and 100% matching of source and reference across conditions.
    if sacrebleu and not args.no_score_recompute:
        add('\n==== INDEPENDENT BLEU / chrF RECOMPUTATION ====')
        for method in ('constant', 'linear'):
            m = warm[method]['test_metrics']
            h = lines_read(root / 'results' / f'full7_{method}_test_full.translations.txt')
            r = lines_read(root / 'results' / f'full7_{method}_test_full.references.txt')
            if h and r and len(h) == len(r):
                bleu = sacrebleu.corpus_bleu(h, [r]).score
                chrf = sacrebleu.corpus_chrf(h, [r]).score
                discrepancy = ((fnum(m.get('bleu')) is not None and abs(bleu - float(m['bleu'])) > 0.05)
                               or (fnum(m.get('chrf')) is not None and abs(chrf - float(m['chrf'])) > 0.05))
                check('WARN' if discrepancy else 'OK', f'{method} score recompute: BLEU={bleu:.3f}, chrF={chrf:.3f}; saved BLEU={m.get("bleu")}, chrF={m.get("chrf")}')
        r1 = lines_read(root / 'results/full7_constant_test_full.references.txt')
        r2 = lines_read(root / 'results/full7_linear_test_full.references.txt')
        if r1 is not None and r2 is not None:
            check('OK' if r1 == r2 else 'FAIL', 'Task 1 full-test references identical and in same order')
        ref = lines_read(data / 'test.en')
        hyp0 = lines_read(lrun / 'zero_shot_test_translations.en')
        hyp1 = lines_read(lrun / 'lora_test_translations.en')
        if ref and hyp0 and hyp1 and len(ref) == len(hyp0) == len(hyp1):
            recomputed = {}
            for variant, h in [('zero_shot', hyp0), ('lora', hyp1)]:
                bleu = sacrebleu.corpus_bleu(h, [ref]).score
                chrf = sacrebleu.corpus_chrf(h, [ref]).score
                m = metrics[variant]
                discrepancy = ((fnum(m.get('BLEU')) is not None and abs(bleu - float(m['BLEU'])) > 0.05)
                               or (fnum(m.get('chrF')) is not None and abs(chrf - float(m['chrF'])) > 0.05))
                check('WARN' if discrepancy else 'OK', f'{variant} recomputed: BLEU={bleu:.3f}, chrF={chrf:.3f} (saved {m.get("BLEU")}, {m.get("chrF")})')
                recomputed[variant] = {'BLEU': bleu, 'chrF': chrf}
            details['task2']['verified_metrics'] = recomputed
            add(f'LoRA ΔBLEU={recomputed["lora"]["BLEU"]-recomputed["zero_shot"]["BLEU"]:+.3f}; ΔchrF={recomputed["lora"]["chrF"]-recomputed["zero_shot"]["chrF"]:+.3f}')
            source = lines_read(data / 'test.sk')
            if source and len(source) == len(ref):
                ranked = []
                for i, (s, r, h0, h1) in enumerate(zip(source, ref, hyp0, hyp1)):
                    z, l = sacrebleu.sentence_chrf(h0, [r]).score, sacrebleu.sentence_chrf(h1, [r]).score
                    ranked.append({'test_index': i, 'source_sk': s, 'reference_en': r,
                                   'zero_shot_en': h0, 'lora_en': h1,
                                   'zero_shot_sentence_chrf': z, 'lora_sentence_chrf': l,
                                   'sentence_chrf_difference': l-z})
                ranked.sort(key=lambda x: x['sentence_chrf_difference'], reverse=True)
                # Selected by sentence chrF deltas; not a substitute for human qualitative judgment.
                chosen = ranked[:8] + ranked[-8:]
                (destination / 'qualitative_candidates.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=2)+'\n',encoding='utf8')
                with (destination / 'qualitative_candidates.csv').open('w', newline='', encoding='utf8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
                    writer.writeheader()
                    writer.writerows(chosen)
                add('Generated 8 largest improvements + 8 largest degradations by sentence chrF; MANUALLY inspect before discussing.')
        else:
            check('WARN', 'Cannot recompute LoRA test metrics; require aligned test.en plus both 2,000-line hypothesis files')

    # Submission audit (manual metadata cannot be inferred).
    add('\n==== SUBMISSION AND REPRODUCIBILITY CHECKLIST ====')
    needed = [
        ('scripts/train_full7.py', 'Task 1 modified training implementation'),
        ('scripts/summarize_full7.py', 'Task 1 convergence plots + table'),
        ('scripts/evaluate_checkpoint.py', 'Task 1 decoding/evaluation'),
        ('scripts/prepare_opus.py', 'Task 2 source-data preparation'),
        ('scripts/lora_engine.py', 'Task 2 LoRA training/evaluation'),
        ('jobs/epoch_full7.sbatch', 'Task 1 Slurm training script'),
        ('jobs/lora_gpu.sbatch', 'Task 2 Slurm training + evaluation script'),
        ('submit_full7.sh', 'Task 1 submission script'),
        ('submit_lora.sh', 'Task 2 submission script'),
        ('tracking/experiment_plan_full7.md', 'Experiment protocol'),
    ]
    for rel, description in needed:
        check('OK' if include(rel) else 'WARN', f'{description}: {rel}')
    for other in ('scripts/preflight_lora_cpu.py', 'jobs/lora_prepare.sbatch', 'jobs/smoke_full7.sbatch',
                  'jobs/eval_full_test_FULL7.sbatch', 'README_FULL7.md', 'README_PREFLIGHT.md',
                  'tracking/toolkit_commit.txt', 'tracking/prepared_dataset_audit.json'):
        include(other)
    for pic in (root / 'tracking').glob('*curve_full7.png'):
        files_for_bundle.add(pic)
    include('tracking/summary_full7.csv')
    if details['toolkit_git'].get('is_git_repository'):
        status = details['toolkit_git'].get('status_porcelain','')
        if status:
            check('WARN', 'Course toolkit has uncommitted changes; record git diff and commit your own fork before submission')
        add('Toolkit commit: ' + str(details['toolkit_git'].get('commit')))
    check('MANUAL', 'PDF must be 2–3 pages INCLUDING figures/tables; include names & OLAT usernames (two members), repository URL')
    check('MANUAL', 'PDF filename uses alphabetically ordered OLAT usernames; one group member submits on OLAT by 2026-11-03 23:59')
    check('MANUAL', 'Report online references/tutorials actually used, data provenance + preprocessing, parameter changes, GPU/training/inference time, results vs baseline, qualitative analysis and reflection')
    check('MANUAL', 'GitHub must contain tested scripts AND improved-model test translations; omit .pt, raw OPUS archives, .venv, private credentials, oversized logs')
    check('MANUAL', 'Only one seed for each method: comparisons are exploratory; do not claim statistical significance without repeat seeds')

    report = {'audit_version': 2, 'root': str(root), 'checks': checks, 'details': details}
    (destination / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    counts = {level: sum(x['status'] == level for x in checks) for level in ('OK','WARN','FAIL','MANUAL')}
    add('\nCOUNTS: '+str(counts))
    add('Interpretation: FAIL=missing/inconsistent evidence, WARN=missing secondary evidence or caveat, MANUAL=human action required.')
    (destination / 'audit.txt').write_text('\n'.join(messages)+'\n', encoding='utf-8')
    summary_rows = []
    for method in ('constant', 'linear'):
        x = warm[method]
        summary_rows.append({'task': '1', 'experiment': method, 'epochs': x['summary'].get('completed_epochs'),
                             'training_seconds_estimate': round(x['total_epoch_process_seconds'], 2),
                             'best_validation_nll': (x['best_validation'] or {}).get('valid_nll_per_token'),
                             'test_n': x['test_metrics'].get('num_sentences'), 'BLEU': x['test_metrics'].get('bleu'),
                             'chrF': x['test_metrics'].get('chrf'), 'inference_seconds': x['test_metrics'].get('seconds'),
                             'training_complete': x['training_completed'], 'test_complete': x['full_test_verified']})
    for variant in ('zero_shot','lora'):
        m=metrics[variant]
        summary_rows.append({'task': '2', 'experiment': variant, 'epochs': 5 if variant == 'lora' and lora_done else None,
                             'training_seconds_estimate': round(total_lora_epoch_time, 2) if variant=='lora' else 0,
                             'best_validation_nll': min((e.get('val_nll') for e in epoch_events if fnum(e.get('val_nll')) is not None), default=None) if variant=='lora' else None,
                             'test_n': m.get('n'), 'BLEU': m.get('BLEU'), 'chrF': m.get('chrF'),
                             'inference_seconds': m.get('seconds'), 'training_complete': lora_done if variant=='lora' else True,
                             'test_complete': m.get('complete')})
    with (destination/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader();writer.writerows(summary_rows)

    for name in ('audit.json', 'audit.txt', 'comparison.csv', 'qualitative_candidates.json', 'qualitative_candidates.csv'):
        f = destination / name
        if f.is_file():
            files_for_bundle.add(f)
    # Capture actual scripts used by this project, not teacher weights or raw corpora.
    evidence_zip = destination / 'atmt_a3_evidence_v2.zip'
    with zipfile.ZipFile(evidence_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(files_for_bundle):
            if file.is_file() and file != evidence_zip:
                z.write(file, arcname=file.relative_to(root))
    print('\n'.join(messages))
    print(f'\nSaved detailed audit: {destination / "audit.txt"}')
    print(f'Saved comparison CSV: {destination / "comparison.csv"}')
    print(f'Saved evidence bundle: {evidence_zip} ({evidence_zip.stat().st_size / 1024**2:.2f} MiB)')
    if counts['FAIL']:
        print(f'WARNING: {counts["FAIL"]} checks failed. Do not claim the experiment/report is complete yet.')
    else:
        print('No critical FAIL checks; complete all WARN/MANUAL items before submitting.')


if __name__ == '__main__':
    main()

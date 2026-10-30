#!/usr/bin/env python3
"""Create Assignment 3 comparison tables and optional plots from recorded runs."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'tracking';OUTPUT.mkdir(exist_ok=True)
rows=[]
for method in ('constant','linear'):
    run=ROOT/'runs'/f'full7_{method}_s42'
    config=run/'config.json'
    if not config.exists():
        print('No run yet:',run)
        continue
    c=json.loads(config.read_text())
    step_file=run/'train_steps.csv'
    steps=list(csv.DictReader(step_file.open())) if step_file.exists() else []
    events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
    validations=[x for x in events if x['type']=='validation']
    summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {}
    metricfile=ROOT/'results'/f'full7_{method}_test_full.metrics.json'
    metrics=json.loads(metricfile.read_text()) if metricfile.exists() else {}
    row={'method':method,'seed':c['seed'],'init_weights_sha256':c.get('initial_weights_sha256',''),
         'configured_max_steps':c['max_steps'],'last_logged_step':steps[-1]['step'] if steps else 0,
         'finished':bool(summary),'lr':c['lr'],'warmup_steps':c['warmup_steps'],
         'best_valid_nll_token':min((x['valid_nll_per_token'] for x in validations),default=''),
         'best_valid_ppl':min((x['valid_ppl'] for x in validations),default=''),
         'completed_epochs':summary.get('completed_epochs',''),
         'train_elapsed_seconds':summary.get('process_seconds',''),
         'examples_seen':summary.get('examples_seen_total',''),
         'valid256_bleu':metrics.get('bleu',''),'valid256_chrf':metrics.get('chrf',''),
         'valid256_inference_seconds':metrics.get('seconds','')}
    rows.append(row)
    print(json.dumps(row,indent=2))
if rows:
    with (OUTPUT/'summary_full7.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved',OUTPUT/'summary_full7.csv')
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not installed; tables generated but plots omitted.')
else:
    for key, y_label, file in [('lr','Learning rate','lr_curve_full7.png'),
                               ('train_nll_per_token','Training NLL per token (logged batches)','train_nll_curve_full7.png')]:
        plt.figure(figsize=(7,4))
        has_data=False
        for method in ('constant','linear'):
            file_path=ROOT/'runs'/f'full7_{method}_s42'/'train_steps.csv'
            if not file_path.exists():continue
            records=list(csv.DictReader(file_path.open()))
            if records:
                plt.plot([int(r['step']) for r in records],[float(r[key]) for r in records],label=method,linewidth=1)
                has_data=True
        if has_data:
            plt.xlabel('Optimizer step');plt.ylabel(y_label);plt.legend();plt.tight_layout();plt.savefig(OUTPUT/file,dpi=160)
            print('Saved',OUTPUT/file)
        plt.close()
    plt.figure(figsize=(7,4));has_data=False
    for method in ('constant','linear'):
        eventfile=ROOT/'runs'/f'full7_{method}_s42'/'events.jsonl'
        if not eventfile.exists():continue
        records=[json.loads(x) for x in eventfile.read_text().splitlines()]
        records=[r for r in records if r.get('type')=='validation']
        if records:
            plt.plot([r['step'] for r in records],[r['valid_nll_per_token'] for r in records],marker='o',label=method);has_data=True
    if has_data:
        plt.xlabel('Optimizer step');plt.ylabel('Validation NLL per token');plt.legend();plt.tight_layout();plt.savefig(OUTPUT/'validation_nll_curve_full7.png',dpi=160)
        print('Saved',OUTPUT/'validation_nll_curve_full7.png')
    plt.close()

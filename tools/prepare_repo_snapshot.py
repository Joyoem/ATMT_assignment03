#!/usr/bin/env python3
"""Stage ATMT Assignment 3 source and experiment evidence into an existing GitHub checkout.

Reads original cluster experiment folders; writes ONLY to --dest. Whitelist only:
no checkpoints, raw parallel corpora, tokenizers, virtual environments or secrets.
Does NOT run git add/commit/push. Review `git status` before publishing.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED = 'joyoem/atmt_assignment03'
MAX_BYTES = 30 * 1024 * 1024
SOURCE = [
    'submit_full7.sh', 'submit_lora.sh',
    'scripts/train_full7.py', 'scripts/summarize_full7.py',
    'scripts/evaluate_checkpoint.py', 'scripts/prepare_opus.py',
    'scripts/preflight_lora_cpu.py', 'scripts/lora_engine.py',
    'jobs/smoke_full7.sbatch', 'jobs/epoch_full7.sbatch',
    'jobs/eval_full_test_FULL7.sbatch', 'jobs/lora_prepare.sbatch',
    'jobs/lora_gpu.sbatch',
    'tracking/experiment_plan_full7.md',
    'tracking/toolkit_commit.txt', 'tracking/prepared_dataset_audit.json',
    'tracking/summary_full7.csv',
    'tracking/audit_v2/audit.txt', 'tracking/audit_v2/audit.json',
    'tracking/audit_v2/comparison.csv',
    'tracking/audit_v2/qualitative_candidates.csv',
    'tracking/audit_v2/qualitative_candidates.json',
    'tracking/lora_cpu_preflight.log',
    'tracking/lora_cpu_preflight_after_data.log',
    'tracking/lora_submission.txt', 'tracking/full7_submitted_jobs.txt',
]
RUN_FILES = [
    'config.json', 'summary.json', 'events.jsonl', 'train_steps.csv',
    'first_batches.jsonl', 'validation_indices.json',
]
LORA_FILES = [
    'config.json', 'steps.csv', 'epochs.jsonl',
    'zero_shot_test_metrics.json', 'lora_test_metrics.json',
    'zero_shot_test_translations.en', 'lora_test_translations.en',
]
TEST_FILES = [
    'full7_constant_test_full.metrics.json',
    'full7_constant_test_full.translations.txt',
    'full7_linear_test_full.metrics.json',
    'full7_linear_test_full.translations.txt',
    'teacher_test_full.metrics.json',
    'teacher_test_full.translations.txt',
]
ALLOW_SUFFIX = {'.py','.sh','.sbatch','.md','.json','.jsonl','.csv','.txt','.log','.png','.en'}
SKIP_NAME = re.compile(r'(private|secret|credentials|\.env$|\.pem$|id_ed25519|\.pt$|\.pth$|\.ckpt$|\.zip$|\.tar$|\.gz$)', re.I)


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    args=ap.parse_args()
    root=args.source.expanduser().resolve()
    dest=args.dest.expanduser().resolve()
    if not (root/'scripts/train_full7.py').is_file() or not (root/'scripts/lora_engine.py').is_file():
        raise SystemExit('ERROR: source does not look like the ATMT A3 cluster experiment directory')
    if not (dest/'.git').is_dir():
        raise SystemExit('ERROR: destination must be a separate cloned Git repository')
    remote=subprocess.run(['git','-C',str(dest),'remote','get-url','origin'],capture_output=True,text=True,check=True).stdout.strip()
    normalized=remote.lower().removesuffix('.git')
    if not (normalized in (f'git@github.com:{EXPECTED}',f'https://github.com/{EXPECTED}',f'ssh://git@github.com/{EXPECTED}')):
        raise SystemExit(f'ERROR: unexpected Git remote {remote!r} — refusing to publish')
    if root==dest or root in dest.parents or dest in root.parents:
        raise SystemExit('ERROR: destination and source must be separate, non-nested directories')

    planned={r:r for r in SOURCE}
    for method in ('constant','linear'):
        for name in RUN_FILES:
            r=f'runs/full7_{method}_s42/{name}'; planned[r]=r
    for name in LORA_FILES:
        r=f'runs/lora_sk_en_r8_a16/{name}';planned[r]=r
    for name in TEST_FILES:
        r=f'results/{name}';planned[r]=r
    for p in (root/'tracking').glob('*curve_full7.png'):
        r=p.relative_to(root).as_posix();planned[r]=r
    planned['data/opus100_en_sk/manifest.json']='evidence/opus100_en_sk_manifest.json'
    # Original model's CLI output is not an independently calibrated benchmark;
    # keep a textual provenance record rather than re-publishing its training corpus.
    copied=[]; missing=[]
    for rel,out in sorted(planned.items()):
        src=root/rel
        if not src.is_file():
            missing.append(rel);continue
        if src.stat().st_size>MAX_BYTES or SKIP_NAME.search(src.name) or src.suffix not in ALLOW_SUFFIX:
            raise SystemExit(f'ERROR: unsafe or unexpectedly large file in allowlist: {rel}')
        target=dest/out
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src,target)
        copied.append({'path':out,'bytes':src.stat().st_size,'sha256':digest(src),'cluster_relative_source':rel})

    # Only publish the original long Slurm logs as summarized evidence. Keep full logs on cluster.
    slurm=[]
    for path in sorted((root/'logs').glob('slurm-*.out')) if (root/'logs').is_dir() else []:
        if not path.is_file(): continue
        lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
        key_pattern=re.compile(r'\b(VALID|EPOCH|JOB_END|ERROR|WARNING|Traceback|RuntimeError|OOM|COMPLETED|FAIL|FINISH|PREPARE_OK)\b')
        selected={*range(min(15,len(lines))), *range(max(0,len(lines)-45),len(lines))}
        important=[i for i,s in enumerate(lines) if key_pattern.search(s)]
        for i in important[-140:]: selected.add(i)
        ordered=sorted(selected)
        # Scrub obvious secret-bearing environment lines if they somehow appear in logs.
        scrub=re.compile(r'(?i)(authorization:|bearer\s+|api[_-]?key\s*=|password\s*=|secret\s*=|ghp_[a-z0-9]+)')
        excerpt=''.join(f'[{i+1}] '+('[REDACTED]\n' if scrub.search(lines[i]) else lines[i]+'\n') for i in ordered)
        output=dest/'evidence/slurm_log_excerpts'/f'{path.stem}.excerpt.txt'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(f'Original: {path.name}; total lines: {len(lines)}; excerpt lines: {len(ordered)}\n'+excerpt,encoding='utf8')
        slurm.append({'filename':path.name,'bytes':path.stat().st_size,'sha256':digest(path),
                      'lines':len(lines),'excerpt_path':output.relative_to(dest).as_posix()})

    versions={'captured_utc':datetime.now(timezone.utc).isoformat(),
              'python_version':sys.version, 'platform':platform.platform(),
              'python_executable':sys.executable, 'packages':{}}
    for module in ('torch','numpy','sentencepiece','sacrebleu','tqdm'):
        try:
            import importlib.metadata
            versions['packages'][module]=importlib.metadata.version(module)
        except Exception as e:
            versions['packages'][module]=f'unavailable: {type(e).__name__}'
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                         capture_output=True,text=True,timeout=5)
        versions['gpu_inventory']=p.stdout.strip() if p.returncode==0 else None
    except Exception:
        versions['gpu_inventory']=None
    (dest/'evidence/runtime_versions.json').write_text(json.dumps(versions,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

    record={'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_experiment_root':str(root),'source_git_repo':'YinZhengxun/atmt_2025',
            'copied_files':copied,'missing_expected_files':missing,
            'slurm_log_summaries':slurm,
            'note':'Missing files are not proof of completion. See tracking/audit_v2/audit.txt. No raw data/weights uploaded.'}
    evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
    (evidence/'upload_manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(f'Copied {len(copied)} safe files; missing {len(missing)} optional/expected files; summarized {len(slurm)} Slurm logs.')
    print('IMPORTANT: Review `git status --short`, audit FAIL/WARN flags and any public data before pushing.')
    if missing:
        print('Missing file examples: '+', '.join(missing[:18]))
    print('No git commit/push was performed. Existing experiments were not modified.')

if __name__=='__main__':main()

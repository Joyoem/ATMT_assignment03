#!/usr/bin/env python3
"""ATMT A3: from-scratch, seven-epoch Czech->English warmup experiment.

Relies on the unmodified 2025 ATMT toolkit's Transformer and pickled data loader.
No teacher checkpoint is loaded for training. CUDA required for Slurm runs.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
TOOLKIT = ROOT / 'toolkit'
sys.path.insert(0, str(TOOLKIT))
from seq2seq import models, utils
from seq2seq.data.dataset import Seq2SeqDataset


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['constant', 'linear'], required=True)
    p.add_argument('--data', default=str(ROOT/'data/extracted/cz-en/prepared'))
    p.add_argument('--src-tokenizer', default=str(ROOT/'data/extracted/cz-en/tokenizers/cz-bpe-8000.model'))
    p.add_argument('--tgt-tokenizer', default=str(ROOT/'data/extracted/cz-en/tokenizers/en-bpe-8000.model'))
    p.add_argument('--run-dir', required=True)
    p.add_argument('--max-steps', type=int, default=0, help='0 means run all epochs; >0 only for smoke testing')
    p.add_argument('--max-epochs', type=int, default=7)
    p.add_argument('--stop-after-epoch', type=int, default=7, help='Number of full epochs completed by end of this job')
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--warmup-start-factor', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--valid-sentences', type=int, default=512)
    p.add_argument('--valid-batch-size', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--clip-norm', type=float, default=4.0)
    p.add_argument('--max-seq-len', type=int, default=300)
    p.add_argument('--dim-embedding', type=int, default=256)
    p.add_argument('--attention-heads', type=int, default=4)
    p.add_argument('--dim-feedforward-encoder', type=int, default=1024)
    p.add_argument('--dim-feedforward-decoder', type=int, default=1024)
    p.add_argument('--n-encoder-layers', type=int, default=3)
    p.add_argument('--n-decoder-layers', type=int, default=3)
    p.add_argument('--encoder-dropout', type=float, default=0.1)
    p.add_argument('--decoder-dropout', type=float, default=0.1)
    p.add_argument('--resume', action='store_true', help='Resume at latest saved checkpoint')
    p.add_argument('--allow-cpu', action='store_true', help='Only for local development, not overnight job')
    args = p.parse_args()
    args.source_lang = 'cz'; args.target_lang = 'en'; args.arch = 'transformer'
    args.encoder_embed_path = None; args.decoder_embed_path = None
    args.cuda = torch.cuda.is_available()
    args.save_dir = str(Path(args.run_dir).resolve())
    args.restore_file = 'checkpoint_last.pt'
    args.max_length = args.max_seq_len
    if not args.cuda and not args.allow_cpu:
        p.error('No CUDA GPU visible: this job needs a GPU allocation; do not run on a login node.')
    if args.max_steps < 0 or args.warmup_steps < 1 or (args.max_steps and args.warmup_steps >= args.max_steps):
        p.error('Require warmup_steps >= 1 and, for smoke mode, warmup_steps < max_steps')
    if not 1 <= args.stop_after_epoch <= args.max_epochs:
        p.error('Require 1 <= stop-after-epoch <= max-epochs')
    if not (0 < args.warmup_start_factor <= 1):
        p.error('warmup-start-factor must be in (0, 1]')
    if args.batch_size < 1 or args.valid_sentences < 1 or args.max_tokens < 1:
        p.error('batch-size, valid-sentences, max-tokens must be positive')
    return args


def lr_for_step(method, step, base_lr, warmup_steps, start_factor):
    """Step starts at 1. Constant plateau vs linear ramp, same initial & final LR."""
    if step > warmup_steps:
        return base_lr
    if method == 'constant':
        return base_lr * start_factor
    if warmup_steps == 1:
        return base_lr
    fraction = (step - 1) / (warmup_steps - 1)
    return base_lr * (start_factor + (1 - start_factor) * fraction)


class SeededBatchSampler:
    """Uniform random sentence sampling, then length bucketing within shuffled pools.

    Unlike the toolkit's global length sort, the first N steps cover a random
    portion of all sentence lengths; important for fixed-step training budgets.
    """
    def __init__(self, dataset, seed, batch_size, max_tokens, pool_size=4096):
        self.dataset=dataset; self.seed=seed; self.batch_size=batch_size
        self.max_tokens=max_tokens; self.pool_size=pool_size

    def __iter__(self):
        rng=np.random.default_rng(self.seed)
        order=rng.permutation(len(self.dataset))
        # DataLoader consumes all batches in order, reproducible per epoch.
        lengths=np.maximum(self.dataset.src_sizes,self.dataset.tgt_sizes)
        for begin in range(0,len(order),self.pool_size):
            pool=order[begin:begin+self.pool_size]
            pool=pool[np.argsort(lengths[pool], kind='stable')]
            batches=[];batch=[];largest=0
            for idx in pool:
                sentence_len=int(lengths[idx]); next_largest=max(largest,sentence_len)
                if batch and (len(batch)>=self.batch_size or next_largest*(len(batch)+1)>self.max_tokens):
                    batches.append(batch);batch=[];largest=0
                batch.append(int(idx));largest=max(largest,sentence_len)
            if batch:batches.append(batch)
            rng.shuffle(batches)
            yield from batches

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def append_jsonl(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        f.flush()


def save_checkpoint(run_dir, args, model, optimizer, step, epoch, epoch_step, val_loss, best_val_loss, examples_seen=0, target_tokens_seen=0):
    """Toolkit-compatible checkpoint plus global-step resume metadata."""
    state = {
        'epoch': epoch, 'last_epoch': epoch, 'val_loss': val_loss,
        'best_loss': best_val_loss, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'args': args,
        'global_step': step, 'epoch_step': epoch_step,
        'warmup_method': args.method, 'examples_seen_total': examples_seen, 'target_tokens_seen_total': target_tokens_seen,
        'rng_torch': torch.get_rng_state(), 'rng_numpy': np.random.get_state(),
        'rng_python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['rng_cuda'] = torch.cuda.get_rng_state_all()
    tmp = run_dir/'checkpoint_last.pt.tmp'
    torch.save(state, tmp)
    os.replace(tmp, run_dir/'checkpoint_last.pt')
    if math.isfinite(val_loss) and val_loss <= best_val_loss:
        # Save best only when validation actually improves (caller passes current best).
        tmp_best = run_dir/'checkpoint_best.pt.tmp'
        torch.save(state, tmp_best)
        os.replace(tmp_best, run_dir/'checkpoint_best.pt')


def check_input_paths(args):
    for path in [args.src_tokenizer, args.tgt_tokenizer] + [str(Path(args.data)/f'{split}.{lang}') for split in ('train','valid') for lang in ('cz','en')]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)


def evaluate_loss(model, criterion, subset, collate_fn, make_batch, tgt_pad, device, bs):
    model.eval()
    loader = DataLoader(subset, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_fn)
    total_nll = 0.; total_tokens = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for sample in loader:
            src, trg_in, trg_out, src_mask, trg_mask = make_batch(sample['src_tokens'], sample['tgt_tokens'])
            logits = model(src, src_mask, trg_in, trg_mask)
            total_nll += criterion(logits.reshape(-1, logits.size(-1)), trg_out).item()
            total_tokens += int(trg_out.ne(tgt_pad).sum().item())
    model.train()
    if total_tokens == 0:
        raise RuntimeError('No non-padding validation tokens')
    return total_nll / total_tokens, time.perf_counter() - start, total_tokens


def main():
    args = get_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    check_input_paths(args)
    csv_file = run_dir/'train_steps.csv'
    events = run_dir/'events.jsonl'
    is_resume = args.resume and (run_dir/'checkpoint_last.pt').is_file()
    if not is_resume and (run_dir/'checkpoint_last.pt').exists():
        raise RuntimeError('Run directory contains checkpoint. Use --resume or choose a fresh directory.')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'RUN method={args.method} cuda={args.cuda} max_epochs={args.max_epochs} stop_after_epoch={args.stop_after_epoch} max_steps={args.max_steps} warmup={args.warmup_steps} run_dir={run_dir}', flush=True)
    src_sp = utils.load_tokenizer(args.src_tokenizer)
    tgt_sp = utils.load_tokenizer(args.tgt_tokenizer)
    if src_sp.pad_id() != tgt_sp.pad_id():
        raise ValueError('Toolkit uses src pad ID for both sides; tokenizer pad IDs must match')
    t0 = time.perf_counter()
    train_ds = Seq2SeqDataset(str(Path(args.data)/'train.cz'), str(Path(args.data)/'train.en'), args.src_tokenizer, args.tgt_tokenizer)
    valid_ds = Seq2SeqDataset(str(Path(args.data)/'valid.cz'), str(Path(args.data)/'valid.en'), args.src_tokenizer, args.tgt_tokenizer)
    if len(train_ds.src_dataset) != len(train_ds.tgt_dataset) or len(valid_ds.src_dataset) != len(valid_ds.tgt_dataset):
        raise ValueError('Train/validation source-target sentence count differs')
    print(f'DATA loaded train={len(train_ds)} valid={len(valid_ds)} seconds={time.perf_counter()-t0:.1f}', flush=True)
    # Evenly spaced fixed selection, shared across both experiments.
    indices = np.linspace(0, len(valid_ds)-1, min(args.valid_sentences,len(valid_ds)), dtype=int).tolist()
    subset = Subset(valid_ds, indices)
    (run_dir/'validation_indices.json').write_text(json.dumps(indices), encoding='utf-8')
    model = models.build_model(args, src_sp, tgt_sp)  # FROM RANDOM INITIALIZATION on CPU
    initial_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        initial_hash.update(name.encode('utf-8'))
        initial_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    initial_weight_sha = initial_hash.hexdigest()
    model = model.to(device)
    print(f'MODEL params={sum(p.numel() for p in model.parameters())} initial_sha256={initial_weight_sha}', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_sp.pad_id(), reduction='sum').to(device)
    make_batch = utils.make_batch_input(device=device, pad=src_sp.pad_id(), max_seq_len=args.max_seq_len)
    git_commit = subprocess.run(['git','-C',str(TOOLKIT),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    hardware = {'host':socket.gethostname(),'platform':platform.platform(),'torch':torch.__version__,
                'numpy':np.__version__, 'gpu':torch.cuda.get_device_name(0) if args.cuda else 'cpu',
                'cuda':torch.version.cuda, 'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                'toolkit_commit':git_commit}
    manifest = {**{k:v for k,v in vars(args).items() if isinstance(v,(str,int,float,bool))},
                'hardware':hardware,'initial_weights_sha256':initial_weight_sha,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'source_tokenizer_sha256':sha256(args.src_tokenizer), 'target_tokenizer_sha256':sha256(args.tgt_tokenizer),
                'train_cz_file_bytes':(Path(args.data)/'train.cz').stat().st_size,
                'train_en_file_bytes':(Path(args.data)/'train.en').stat().st_size,
                'warmup_definition':'constant: lr=base_lr*start_factor for steps 1..W, then base_lr; linear: interpolate start_factor*base_lr at step1 to base_lr at step W; then base_lr',
                'model_init':'random from seed; never loaded teacher weights',
                'sampler':'SeededBatchSampler: random permutation and length buckets inside pools of 4096; same seeded sampler for both methods',
                'validation':'fixed evenly-spaced subset, teacher forcing only; at checkpoints AND end of every epoch; no BLEU during training',
                'training_loss_scale':'original toolkit: summed cross-entropy / batch sentence count'}
    if not is_resume:
        (run_dir/'config.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n', encoding='utf-8')
    step=0; epoch=0; epoch_step=0; best_loss=float('inf'); last_val=float('inf')
    resume_rng = None
    examples_seen=0;target_tokens_seen=0
    last_validation_step=-1
    if is_resume:
        checkpoint = torch.load(run_dir/'checkpoint_last.pt', map_location='cpu',weights_only=False)
        if checkpoint['warmup_method'] != args.method or checkpoint['args'].seed != args.seed:
            raise RuntimeError('Resume method/seed mismatch')
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        step=int(checkpoint['global_step']); epoch=int(checkpoint['epoch']); epoch_step=int(checkpoint['epoch_step'])
        best_loss=float(checkpoint['best_loss']);last_val=float(checkpoint['val_loss']);resume_rng=checkpoint
        examples_seen=int(checkpoint.get('examples_seen_total',0));target_tokens_seen=int(checkpoint.get('target_tokens_seen_total',0))
        last_validation_step=step
        print(f'RESUMING at global_step={step}, epoch={epoch}, epoch_step={epoch_step}',flush=True)
    fieldnames=['step','epoch','step_in_epoch','method','lr','train_loss_per_sentence','train_nll_per_token','grad_norm','batch_size','target_tokens','step_seconds','elapsed_seconds','gpu_peak_mb']
    if not csv_file.exists():
        with csv_file.open('w',newline='') as f:
            csv.DictWriter(f,fieldnames=fieldnames).writeheader()
    total_start=time.perf_counter()
    interval_sum=0.; interval_tokens=0;interval_steps=0
    while epoch < args.stop_after_epoch and (args.max_steps == 0 or step < args.max_steps):
        # Deterministic and uniform corpus sampling for both methods.
        sampler=SeededBatchSampler(train_ds,args.seed+epoch,args.batch_size,args.max_tokens)
        loader=DataLoader(train_ds,num_workers=0,collate_fn=train_ds.collater,batch_sampler=sampler)
        for batch_i, sample in enumerate(loader):
            if batch_i < epoch_step:
                continue
            if not is_resume and batch_i < 10 and epoch == 0:
                append_jsonl(run_dir/'first_batches.jsonl',
                             {'batch_i':batch_i,'sample_ids':[int(x) for x in sample['id'].tolist()]})
            if resume_rng is not None:
                # Restoring model RNG after skipping the sampler's already-seen batches.
                torch.set_rng_state(resume_rng['rng_torch'])
                if args.cuda and 'rng_cuda' in resume_rng:
                    torch.cuda.set_rng_state_all(resume_rng['rng_cuda'])
                random.setstate(resume_rng['rng_python'])
                np.random.set_state(resume_rng['rng_numpy'])
                resume_rng=None
            tick=time.perf_counter()
            model.train()
            src,trg_in,trg_out,src_mask,trg_mask=make_batch(sample['src_tokens'],sample['tgt_tokens'])
            lr=lr_for_step(args.method,step+1,args.lr,args.warmup_steps,args.warmup_start_factor)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            logits=model(src,src_mask,trg_in,trg_mask)
            loss_sum=criterion(logits.reshape(-1,logits.size(-1)),trg_out)
            loss=loss_sum/len(sample['src_lengths'])  # match toolkit original training objective
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step+1}: {loss.item()}')
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'Non-finite gradient norm at step {step+1}: {grad_norm.item()}')
            optimizer.step()
            step+=1;epoch_step=batch_i+1
            target_tokens=int(trg_out.ne(tgt_sp.pad_id()).sum().item())
            dt=time.perf_counter()-tick
            loss_per_sentence=float(loss.item());loss_per_token=float(loss_sum.item())/max(target_tokens,1)
            interval_sum+=dt;interval_tokens+=target_tokens;interval_steps+=1
            examples_seen+=len(sample['src_lengths']);target_tokens_seen+=target_tokens
            if step == 1 or step % args.log_every == 0 or (args.max_steps > 0 and step == args.max_steps):
                row={'step':step,'epoch':epoch,'step_in_epoch':epoch_step,'method':args.method,
                     'lr':lr,'train_loss_per_sentence':loss_per_sentence,'train_nll_per_token':loss_per_token,
                     'grad_norm':float(grad_norm.item()),'batch_size':len(sample['src_lengths']),
                     'target_tokens':target_tokens,'step_seconds':dt,'elapsed_seconds':time.perf_counter()-total_start,
                     'gpu_peak_mb':round(torch.cuda.max_memory_allocated()/1024**2,1) if args.cuda else 0}
                with csv_file.open('a',newline='') as f:
                    csv.DictWriter(f,fieldnames=fieldnames).writerow(row)
                print(f'STEP={step}/{args.max_steps} method={args.method} lr={lr:.8g} nll/token={loss_per_token:.4f} sec/step_recent={interval_sum/max(interval_steps,1):.3f} tokens/sec_recent={interval_tokens/max(interval_sum,1e-6):.0f}',flush=True)
                interval_sum=0.;interval_tokens=0;interval_steps=0
            if step % args.eval_every == 0 or (args.max_steps > 0 and step == args.max_steps):
                val,evalsecs,validtokens=evaluate_loss(model,criterion,subset,valid_ds.collater,make_batch,tgt_sp.pad_id(),device,args.valid_batch_size)
                improved=val < best_loss
                if improved:best_loss=val
                last_val=val
                record={'type':'validation','step':step,'epoch':epoch,'valid_nll_per_token':val,
                        'valid_ppl':float(math.exp(min(val,50.))), 'valid_tokens':validtokens,
                        'valid_sentences':len(subset),'validation_seconds':evalsecs,
                        'examples_seen_since_process_start':examples_seen,'target_tokens_since_process_start':target_tokens_seen,
                        'elapsed_seconds':time.perf_counter()-total_start,'new_best':improved}
                append_jsonl(events,record)
                # Always save the latest; save best only when improved.
                save_checkpoint(run_dir,args,model,optimizer,step,epoch,epoch_step,val,best_loss,examples_seen,target_tokens_seen)
                last_validation_step=step
                print(f'VALID step={step} nll/token={val:.4f} ppl={math.exp(min(val,50)):.3f} seconds={evalsecs:.1f} improved={improved}',flush=True)
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break  # smoke mode may finish mid-epoch
        # A full pass through the entire two-million-pair dataset is complete.
        if last_validation_step != step:
            val,evalsecs,validtokens=evaluate_loss(model,criterion,subset,valid_ds.collater,make_batch,tgt_sp.pad_id(),device,args.valid_batch_size)
            improved=val < best_loss
            if improved: best_loss=val
            last_val=val
            append_jsonl(events,{'type':'validation','step':step,'epoch':epoch,'valid_nll_per_token':val,
                'valid_ppl':float(math.exp(min(val,50.))), 'valid_tokens':validtokens,
                'valid_sentences':len(subset),'validation_seconds':evalsecs,
                'examples_seen_total':examples_seen,'target_tokens_seen_total':target_tokens_seen,
                'elapsed_seconds':time.perf_counter()-total_start,'new_best':improved})
            save_checkpoint(run_dir,args,model,optimizer,step,epoch,epoch_step,val,best_loss,examples_seen,target_tokens_seen)
            last_validation_step=step
            print(f'EPOCH VALID epoch={epoch} step={step} nll/token={val:.4f} ppl={math.exp(min(val,50)):.3f}',flush=True)
        append_jsonl(events, {'type':'epoch_completed','completed_epoch_index':epoch,'completed_epochs':epoch+1,
                              'step':step,'examples_seen_total':examples_seen,
                              'target_tokens_seen_total':target_tokens_seen,
                              'process_seconds':time.perf_counter()-total_start})
        print(f'END epoch={epoch} completed_epochs={epoch+1} global_step={step} examples_seen_total={examples_seen}',flush=True)
        epoch+=1;epoch_step=0
        if epoch < args.max_epochs:
            # At the epoch boundary, prepare last checkpoint to resume directly
            # from the NEXT epoch without replaying one epoch's mini-batches.
            latest = torch.load(run_dir/'checkpoint_last.pt', map_location='cpu', weights_only=False)
            latest['epoch']=epoch
            latest['epoch_step']=0
            latest['rng_torch']=torch.get_rng_state()
            latest['rng_numpy']=np.random.get_state()
            latest['rng_python']=random.getstate()
            if args.cuda: latest['rng_cuda']=torch.cuda.get_rng_state_all()
            tmp = run_dir/'checkpoint_last.pt.tmp'
            torch.save(latest,tmp);os.replace(tmp,run_dir/'checkpoint_last.pt')
        if epoch >= args.stop_after_epoch:
            break
    finish={'type':'completed' if (epoch >= args.max_epochs or args.max_steps > 0) else 'paused_after_epoch',
            'step':step, 'completed_epochs':epoch,
            'process_seconds':time.perf_counter()-total_start,
            'best_valid_nll_per_token':best_loss,'last_valid_nll_per_token':last_val,
            'examples_seen_total':examples_seen,'target_tokens_seen_total':target_tokens_seen}
    append_jsonl(events,finish)
    if finish['type']=='completed':
        (run_dir/'summary.json').write_text(json.dumps({**finish,'method':args.method,'seed':args.seed,'lr':args.lr,
                                       'warmup_steps':args.warmup_steps,'max_epochs':args.max_epochs},indent=2)+'\n')
    print('JOB_END '+json.dumps(finish),flush=True)

if __name__=='__main__':
    main()

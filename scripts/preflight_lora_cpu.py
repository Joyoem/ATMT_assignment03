"""Lightweight CPU preflight for ATMT LoRA; no GPU or Slurm job required.
Usage: python scripts/preflight_lora_cpu.py
"""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path.home() / 'projects' / 'atmt_a3'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'toolkit'))
from seq2seq import models, utils
from lora_engine import load_model, LoRALinear, nll


def assert_example_archive_extraction():
    """Prove the fix matches OPUS-100's official file naming, offline."""
    with tempfile.TemporaryDirectory(prefix='atmt_lora_preflight_') as td:
        tmp = Path(td)
        archive = tmp / 'fixture.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for split in ('train', 'dev', 'test'):
                for lang in ('sk', 'en'):
                    name = f'opus-100-corpus/v1.0/supervised/en-sk/opus.en-sk-{split}.{lang}'
                    values = [f'{split} {lang} sentence {i}' for i in range(3)]
                    content = ('\n'.join(values) + '\n').encode('utf-8')
                    ti = tarfile.TarInfo(name)
                    ti.size = len(content)
                    tar.addfile(ti, io.BytesIO(content))
        cmd = [sys.executable, str(ROOT / 'scripts' / 'prepare_opus.py'), '--root', str(tmp),
               '--archive', str(archive), '--train-size', '2', '--seed', '42']
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        manifest = json.loads((tmp / 'data/opus100_en_sk/manifest.json').read_text())
        assert manifest['counts']['train']['selected_pairs'] == 2
        assert manifest['counts']['dev']['selected_pairs'] == 3
        assert manifest['counts']['test']['selected_pairs'] == 3
    print('PASS offline archive-format fixture', flush=True)


def main():
    assert_example_archive_extraction()
    base = ROOT / 'data/extracted/cz-en'
    teacher = base / 'checkpoints/checkpoint_best.pt'
    print('Model checkpoint:', teacher)
    src = utils.load_tokenizer(str(base / 'tokenizers/cz-bpe-8000.model'))
    tgt = utils.load_tokenizer(str(base / 'tokenizers/en-bpe-8000.model'))
    model, wrapped, ck = load_model(str(teacher), src, tgt, torch.device('cpu'), 8, 16)
    assert wrapped == 18
    model.eval()
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n,p in trainable)
    frozen = [n for n,p in model.named_parameters() if not p.requires_grad]
    tr = sum(p.numel() for _,p in trainable)
    total = sum(p.numel() for p in model.parameters())
    print(f'LoRA modules={wrapped}; trainable={tr:,}; total={total:,}; share={tr / total:.3%}; frozen={len(frozen)}')

    srcids = src.encode('Toto je maly slovensky test.', out_type=int)[:24] + [src.eos_id()]
    tgtids = tgt.encode('This is a small Slovak test.', out_type=int)[:24] + [tgt.eos_id()]
    S = torch.tensor([srcids], dtype=torch.long)
    T = torch.tensor([tgtids], dtype=torch.long)
    with torch.no_grad():
        state = ck['model']
        baseline = models.build_model(ck['args'], src, tgt)
        baseline.load_state_dict(state, strict=True)
        baseline.eval()
        tgtin = torch.cat([torch.tensor([[tgt.bos_id()]]), T[:,:-1]], dim=1)
        mask = (S == src.pad_id())[:,None,None,:]
        tmask = (tgtin == tgt.pad_id())[:,None,None,:]
        b = baseline(S, mask, tgtin, tmask)
        l = model(S, mask, tgtin, tmask)
        assert torch.allclose(b, l, rtol=1e-5, atol=1e-5), (b-l).abs().max()
    print('PASS initial LoRA output equals unmodified baseline', flush=True)
    del baseline, b, l

    model.train()
    first_base = model.encoder.tok_embed.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for _,p in trainable], lr=3e-4)
    loss, ntok = nll(model, (S,T), src.pad_id(), tgt.pad_id(), torch.device('cpu'))
    assert torch.isfinite(loss).item()
    loss.backward()
    grad_b = [(n,p.grad) for n,p in trainable if n.endswith('lora_B')]
    assert grad_b and any(g is not None and bool(torch.any(g != 0)) for _,g in grad_b)
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
    optimizer.step()
    assert torch.equal(first_base, model.encoder.tok_embed.weight)
    print(f'PASS CPU backward/optimizer update; loss={loss.item():.4f}; target_tokens={ntok}', flush=True)

    data = ROOT / 'data/opus100_en_sk'
    if (data / 'manifest.json').exists():
        manifest = json.loads((data/'manifest.json').read_text())
        for split in ('train','dev','test'):
            sk = (data/f'{split}.sk').read_text(encoding='utf8').splitlines()
            en = (data/f'{split}.en').read_text(encoding='utf8').splitlines()
            n = manifest['counts'][split]['selected_pairs']
            assert len(sk) == len(en) == n, (split,len(sk),len(en),n)
            print(f'PASS {split} aligned counts: {n:,}')
        unknown = src.unk_id()
        sample = (data/'train.sk').read_text(encoding='utf8').splitlines()[:1000]
        pieces = [src.encode(s,out_type=int) for s in sample]
        tokens = sum(len(p) for p in pieces)
        unk = sum(sum(t==unknown for t in p) for p in pieces)
        print(f'Czech tokenizer on Slovak: UNK/token={unk}/{tokens} ({unk/max(tokens,1):.2%}); sample={len(sample)}')
    else:
        print('NOTICE prepared OPUS data not present yet: only fixture+model checks ran')
    print('CPU_PREFLIGHT_OK')


if __name__ == '__main__':
    main()

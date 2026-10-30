#!/usr/bin/env python3
"""Fair, fixed-subset greedy BLEU check for teacher and new checkpoints.

For full test output, pass --split test --limit 0. Uses the toolkit's original
Transformer greedy decode for every model (NOT Beam Search).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'toolkit'))
from seq2seq import models, utils
from seq2seq.data.dataset import Seq2SeqDataset
from seq2seq.decode import decode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=['valid','test'],default='valid')
    p.add_argument('--limit',type=int,default=256,help='0 = all; otherwise evenly-spaced fixed subset')
    p.add_argument('--max-len',type=int,default=100)
    p.add_argument('--output-prefix',required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():
        p.error('Evaluation requires a GPU allocation; submit with sbatch.')
    device=torch.device('cuda')
    data=ROOT/'data/extracted/cz-en'
    src_path=str(data/'tokenizers/cz-bpe-8000.model')
    tgt_path=str(data/'tokenizers/en-bpe-8000.model')
    src=utils.load_tokenizer(src_path);tgt=utils.load_tokenizer(tgt_path)
    ckpt=torch.load(args.checkpoint, map_location='cpu',weights_only=False)
    arch_args=ckpt['args'];arch_args.cuda=True
    model=models.build_model(arch_args,src,tgt).to(device)
    model.load_state_dict(ckpt['model'],strict=True);model.eval()
    maxlen=min(args.max_len,model.decoder.pos_embed.size(1))
    ds=Seq2SeqDataset(str(data/f'prepared/{args.split}.cz'),str(data/f'prepared/{args.split}.en'),src_path,tgt_path)
    indices=range(len(ds)) if args.limit==0 else np.linspace(0,len(ds)-1,min(args.limit,len(ds)),dtype=int).tolist()
    prefix=Path(args.output_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    hyps=[];refs=[];records=[]
    t0=time.perf_counter()
    with torch.inference_mode():
        for i,index in enumerate(indices):
            sample=ds[index]
            src_ids=sample['source'].unsqueeze(0).to(device)
            src_pad_mask=(src_ids==src.pad_id()).view(1,1,1,-1)
            predictions=decode(model,src_ids,src_pad_mask,maxlen,tgt,arch_args,device)
            ids=predictions[0]
            if tgt.eos_id() in ids:ids=ids[:ids.index(tgt.eos_id())]
            hyp=tgt.decode([int(x) for x in ids if int(x)!=tgt.pad_id()])
            ref_ids=sample['target'].tolist()
            if tgt.eos_id() in ref_ids:ref_ids=ref_ids[:ref_ids.index(tgt.eos_id())]
            ref=tgt.decode([int(x) for x in ref_ids if int(x)!=tgt.pad_id()])
            hyps.append(hyp);refs.append(ref)
            if len(records)<50:records.append({'index':int(index),'hypothesis':hyp,'reference':ref})
            if (i+1)%100==0:print(f'GEN {i+1}/{len(indices)}',flush=True)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-t0
    bleu=sacrebleu.corpus_bleu(hyps,[refs])
    chrf=sacrebleu.corpus_chrf(hyps,[refs])
    (prefix.with_suffix('.translations.txt')).write_text('\n'.join(hyps)+'\n',encoding='utf-8')
    (prefix.with_suffix('.references.txt')).write_text('\n'.join(refs)+'\n',encoding='utf-8')
    result={'checkpoint':str(Path(args.checkpoint).resolve()),'split':args.split,'limit':args.limit,
            'num_sentences':len(hyps),'max_len':maxlen,'decoding':'toolkit_greedy',
            'bleu':bleu.score,'chrf':chrf.score,'seconds':elapsed,'sentences_per_second':len(hyps)/max(elapsed,1e-6),
            'bleu_signature':str(bleu.get_signature()) if hasattr(bleu,'get_signature') else None,
            'sample_records':records}
    (prefix.with_suffix('.metrics.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULT '+json.dumps({k:v for k,v in result.items() if k!='sample_records'}),flush=True)

if __name__=='__main__':
    main()

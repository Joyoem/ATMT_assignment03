import argparse,csv,hashlib,json,random,sys,time,subprocess
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
ROOT=Path.home()/'projects/atmt_a3';sys.path.insert(0,str(ROOT/'toolkit'))
from seq2seq import models,utils
from seq2seq.data.dataset import Seq2SeqDataset

class LoRALinear(nn.Module):
    def __init__(self,base,r=8,alpha=16):
        super().__init__();self.base=base;self.r=r;self.scaling=alpha/r
        self.lora_A=nn.Parameter(torch.empty(r,base.in_features));self.lora_B=nn.Parameter(torch.zeros(base.out_features,r))
        nn.init.kaiming_uniform_(self.lora_A,a=5**.5)
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.lora_A),self.lora_B)*self.scaling

def attach_lora(model,r,alpha):
    n=0
    for module in list(model.modules()):
        for name in ('WQ','WV'):
            child=getattr(module,name,None)
            if isinstance(child,nn.Linear):setattr(module,name,LoRALinear(child,r,alpha));n+=1
    assert n==18,f'Expected Q/V on 9 attention modules x2, got {n}'
    for name,p in model.named_parameters():p.requires_grad=('lora_A' in name or 'lora_B' in name)
    return n

def load_model(checkpoint,src,tgt,device,rank,alpha):
    # trusted teacher provided by course; pickle executes code
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    args=ck['args'];args.cuda=device.type=='cuda'
    model=models.build_model(args,src,tgt);model.load_state_dict(ck['model'],strict=True)
    n=attach_lora(model,rank,alpha)
    return model.to(device),n,ck

class Parallel(Dataset):
    def __init__(self,data,split,src,tgt,maxlen):
        sk=(data/f'{split}.sk').read_text(encoding='utf8').splitlines();en=(data/f'{split}.en').read_text(encoding='utf8').splitlines();assert len(sk)==len(en)
        self.samples=[];self.raw=[]
        for s,t in zip(sk,en):
            a=src.encode(s,out_type=int)[:maxlen-1]+[src.eos_id()]
            b=tgt.encode(t,out_type=int)[:maxlen-1]+[tgt.eos_id()]
            self.samples.append((a,b));self.raw.append((s,t))
        assert self.samples
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):return self.samples[i]

def collate(items,srcpad,tgtpad):
    s,t=zip(*items);S=torch.full((len(s),max(map(len,s))),srcpad,dtype=torch.long);T=torch.full((len(t),max(map(len,t))),tgtpad,dtype=torch.long)
    for i,(a,b) in enumerate(zip(s,t)):S[i,:len(a)]=torch.tensor(a);T[i,:len(b)]=torch.tensor(b)
    return S,T

def makebatch(S,T,srcpad,tgtpad,device):
    S=S.to(device);T=T.to(device);trgin=torch.cat([torch.full((len(T),1),1,device=device,dtype=torch.long),T[:,:-1]],dim=1)
    return S,trgin,T,(S==srcpad)[:,None,None,:],(trgin==tgtpad)[:,None,None,:]

def nll(model,batch,srcpad,tgtpad,device):
    S,T=batch;S,inp,labels,sm,tm=makebatch(S,T,srcpad,tgtpad,device)
    logits=model(S,sm,inp,tm)
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=tgtpad,reduction='sum')
    nt=(labels!=tgtpad).sum().clamp_min(1)
    return loss/nt,nt.item()

def val(model,dl,sp,tp,device):
    model.eval();total=0;tokens=0
    with torch.inference_mode():
        for batch in dl:
            loss,n=nll(model,batch,sp,tp,device);total+=loss.item()*n;tokens+=n
    return total/tokens

def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str))

def train(a):
    assert torch.cuda.is_available(),'Slurm GPU allocation required';device=torch.device('cuda')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    data=ROOT/'data/opus100_en_sk'; run=ROOT/a.run_dir;run.mkdir(parents=True,exist_ok=True)
    src=utils.load_tokenizer(str(ROOT/'data/extracted/cz-en/tokenizers/cz-bpe-8000.model'))
    tgt=utils.load_tokenizer(str(ROOT/'data/extracted/cz-en/tokenizers/en-bpe-8000.model'))
    teacher=str(ROOT/'data/extracted/cz-en/checkpoints/checkpoint_best.pt')
    ds=Parallel(data,'train',src,tgt,a.max_len);dv=Parallel(data,'dev',src,tgt,a.max_len)
    if a.train_limit:ds.samples=ds.samples[:a.train_limit];ds.raw=ds.raw[:a.train_limit]
    if a.valid_limit:dv.samples=dv.samples[:a.valid_limit];dv.raw=dv.raw[:a.valid_limit]
    model,n,ck=load_model(teacher,src,tgt,device,a.rank,a.alpha)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);total=sum(p.numel() for p in model.parameters());assert trainable<total
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    config=dict(vars(a),train_sentences=len(ds),valid_sentences=len(dv),teacher_epoch=ck['epoch'],teacher_val_loss=ck['val_loss'],total_parameters=total,trainable_parameters=trainable,modules_wrapped=n,torch_version=torch.__version__,cuda=torch.version.cuda,opus_manifest=json.loads((data/'manifest.json').read_text()),toolkit_commit=subprocess.getoutput(f'git -C {ROOT}/toolkit rev-parse HEAD'),gpu=torch.cuda.get_device_name())
    write_json(run/'config.json',config)
    step=0;best=float('inf');start_epoch=0
    latest=run/'checkpoint_last.pt'
    if a.resume:
        assert latest.exists(),f'Missing resume checkpoint: {latest}'
        state=torch.load(latest,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);step=state['step'];best=state['best'];start_epoch=state['epoch_completed'];print('RESUMED',start_epoch,step,flush=True)
    assert start_epoch<a.stop_after_epoch
    dlv=DataLoader(dv,batch_size=a.valid_batch,shuffle=False,collate_fn=lambda xs:collate(xs,src.pad_id(),tgt.pad_id()),num_workers=0)
    csvfile=run/'steps.csv';exists=csvfile.exists();f=open(csvfile,'a',newline='');writer=csv.writer(f)
    if not exists:writer.writerow(['epoch','step','lr','train_nll_per_token','target_tokens','step_seconds','peak_gpu_MB'])
    for epoch in range(start_epoch,a.stop_after_epoch):
        gen=torch.Generator().manual_seed(a.seed+epoch)
        dl=DataLoader(ds,batch_size=a.batch,shuffle=True,generator=gen,collate_fn=lambda xs:collate(xs,src.pad_id(),tgt.pad_id()),num_workers=0)
        model.train();t0=time.perf_counter();running=0;epoch_steps=0
        for batch in dl:
            ts=time.perf_counter();optim.zero_grad(set_to_none=True)
            loss,num=nll(model,batch,src.pad_id(),tgt.pad_id(),device);assert torch.isfinite(loss).item()
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optim.step();step+=1
            running+=loss.item();epoch_steps+=1
            if step%a.log_every==0:
                torch.cuda.synchronize();writer.writerow([epoch,step,a.lr,loss.item(),num,round(time.perf_counter()-ts,4),round(torch.cuda.max_memory_allocated()/2**20,1)]);f.flush()
                print(f'EPOCH={epoch+1} STEP={step} NLL={loss.item():.4f} avg={running/epoch_steps:.4f}',flush=True)
        score=val(model,dlv,src.pad_id(),tgt.pad_id(),device)
        improved=score<best;best=min(score,best)
        state={'model':model.state_dict(),'optimizer':optim.state_dict(),'epoch_completed':epoch+1,'step':step,'best':best,'val_nll':score,'config':config}
        torch.save(state,latest)
        if improved:torch.save(state,run/'checkpoint_best.pt')
        with (run/'epochs.jsonl').open('a') as e:e.write(json.dumps({'epoch_completed':epoch+1,'step':step,'val_nll':score,'val_ppl':float(np.exp(score)),'train_elapsed_seconds':time.perf_counter()-t0,'best':improved})+'\n')
        print('EPOCH_FINISHED',epoch+1,'val_nll',score,'best',improved,flush=True)
    f.close()

def decode_greedy(model,srcids,srcpad,tgt,device,maxgen):
    # one sentence per batch, greedy identical conceptual baseline; avoids toolkit slicing bug at boundary
    s=torch.tensor(srcids,device=device,dtype=torch.long)[None,:];mask=(s==srcpad)[:,None,None,:]
    ids=torch.tensor([[tgt.bos_id()]],device=device);out=[]
    with torch.inference_mode():
        for _ in range(maxgen):
            logits=model(s,mask,ids,(ids==tgt.pad_id())[:,None,None,:]);j=int(logits[0,-1].argmax());
            if j==tgt.eos_id():break
            out.append(j);ids=torch.cat([ids,torch.tensor([[j]],device=device)],dim=1)
    return tgt.decode(out)

def eval_model(a):
    import sacrebleu
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');data=ROOT/'data/opus100_en_sk';run=ROOT/a.run_dir;run.mkdir(parents=True,exist_ok=True)
    src=utils.load_tokenizer(str(ROOT/'data/extracted/cz-en/tokenizers/cz-bpe-8000.model'));tgt=utils.load_tokenizer(str(ROOT/'data/extracted/cz-en/tokenizers/en-bpe-8000.model'))
    model,n,_=load_model(str(ROOT/'data/extracted/cz-en/checkpoints/checkpoint_best.pt'),src,tgt,device,a.rank,a.alpha)
    if a.variant=='lora':
        c=torch.load(run/'checkpoint_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(c['model'])
    model.eval();ds=Parallel(data,a.split,src,tgt,a.max_len)
    hypotheses=[];references=[];t0=time.perf_counter();outfile=run/f'{a.variant}_{a.split}_translations.en'
    with outfile.open('w') as f:
        for i,(s,t) in enumerate(ds.raw):
            prediction=decode_greedy(model,ds.samples[i][0],src.pad_id(),tgt,device,a.max_gen)
            hypotheses.append(prediction);references.append(t);f.write(prediction+'\n')
            if (i+1)%100==0: f.flush();print(a.variant,a.split,i+1,'/',len(ds),flush=True)
    elapsed=time.perf_counter()-t0
    bleu=sacrebleu.corpus_bleu(hypotheses,[references]);chrf=sacrebleu.corpus_chrf(hypotheses,[references])
    report={'variant':a.variant,'split':a.split,'n':len(ds),'BLEU':bleu.score,'chrF':chrf.score,'sacrebleu_signature':str(bleu),'decoding':'greedy, max_gen='+str(a.max_gen),'seconds':elapsed,'sentence_per_second':len(ds)/elapsed}
    write_json(run/f'{a.variant}_{a.split}_metrics.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval']);p.add_argument('--run-dir',default='runs/lora_sk_en_r8_a16');p.add_argument('--train-limit',type=int,default=0);p.add_argument('--valid-limit',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--rank',type=int,default=8);p.add_argument('--alpha',type=float,default=16);p.add_argument('--lr',type=float,default=0.0003);p.add_argument('--batch',type=int,default=64);p.add_argument('--valid-batch',type=int,default=32);p.add_argument('--max-len',type=int,default=96);p.add_argument('--max-gen',type=int,default=95);p.add_argument('--stop-after-epoch',type=int,default=1);p.add_argument('--resume',action='store_true');p.add_argument('--log-every',type=int,default=100);p.add_argument('--variant',choices=['zero_shot','lora'],default='lora');p.add_argument('--split',choices=['dev','test'],default='test');a=p.parse_args()
    train(a) if a.mode=='train' else eval_model(a)

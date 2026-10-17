"""Frozen finite training, independent jobs, causal memory and complete receipts."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,hashlib,json,time,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
import core

ROOT=Path(__file__).resolve().parent

def jwrite(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)

def source_hashes():return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('core.py','experiment.py')}

def fit(p,ref,mem,end,mode,seed,max_epochs=160):
    core.configure();ids,x,y,tr,va,sd=core.support(p,end)
    model=core.make_model(mode,seed)
    bt=mem.tensors(sd,ids[tr],mode);bv=mem.tensors(sd,ids[va],mode)
    yt=torch.tensor(y[tr]/sd);yv=torch.tensor(y[va]/sd)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    with torch.no_grad():best=float(((model(bv)-yv)**2).mean())
    initial_val=best;state=copy.deepcopy(model.state_dict());selected=0;wait=0;trace=[]
    start=time.perf_counter()
    for epoch in range(1,max_epochs+1):
        optimizer.zero_grad();pred,detail=model(bt,True)
        mse=((pred-yt)**2).mean();penalty=detail['correction'].square().mean() if mode!='GRU' else mse*0.
        loss=mse+penalty;loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        optimizer.step()
        with torch.no_grad():vl=float(((model(bv)-yv)**2).mean())
        row=[float(loss.detach()),float(mse.detach()),float(penalty.detach()),vl,gn]
        if not np.isfinite(row).all():raise FloatingPointError(f'Nonfinite {mode}/{seed}/{end}/{epoch}')
        trace.append(row)
        if vl<best-1e-6:best=vl;state=copy.deepcopy(model.state_dict());selected=epoch;wait=0
        else:wait+=1
        if wait>=20:break
    model.load_state_dict(state)
    count=sum(x.numel() for x in model.parameters())
    log=dict(mode=mode,seed=seed,end=end,epochs=len(trace),selected_epoch=selected,seconds=time.perf_counter()-start,
             train_count=len(tr),validation_count=len(va),max_training_target=int(ids[tr[-1]]+20),
             first_validation_origin=int(ids[va[0]]),last_validation_target=int(ids[va[-1]]+20),
             scale=sd,allocated_parameters=count,initial_validation_MSE=initial_val,best_validation_MSE=best,
             memory_support='each_source+20<its_query',trace_columns=['objective','mse','correction_penalty','validation_mse','unclipped_grad_norm'],trace=trace)
    return model,sd,log

def prediction(model,mem,sd,ids):
    b=mem.tensors(sd,ids,model.mode)
    with torch.no_grad():pred,d=model(b,True)
    output=pred.numpy()*sd;extra={}
    for k in ['w','pi','beta','memory','param']:
        if k in d:extra[k]=d[k].numpy()
    extra['correction']=d['correction'].numpy()*sd
    if 'w' in extra:
        extra['source_origins']=mem.candidates['thin' if model.mode=='THIN' else 'regular'][ids]
        extra['valid_mask']=mem.masks['thin' if model.mode=='THIN' else 'regular'][ids]
    return output,extra

def worker(args):
    data,out,mode,seed,block=args;out=Path(out);key=f'{mode}_s{seed}_b{block:02d}'
    dest=out/'fits'/f'{key}.npz';receipt=out/'fits'/f'{key}.json'
    p,dates,audit=core.load_prefix(data);end=459+40*block;ids=np.arange(end,min(end+40,930))
    hashes=source_hashes()
    if dest.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if old['source_sha256']!=hashes:raise ValueError('Resume source mismatch')
        return old
    start=time.perf_counter();ref=core.reference(p);mem=core.MemoryData(p,ref)
    model,sd,log=fit(p,ref,mem,end,mode,seed)
    pred,extras=prediction(model,mem,sd,ids)
    state={f'parameter::{k}':v.detach().numpy() for k,v in model.state_dict().items()}
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,origins=ids,prediction=pred,**extras,**state)
    if 'w' in extras:
        log['weight_diagnostics']=dict(negative_fraction=float(np.mean(extras['w']< -1e-10)),min=float(extras['w'].min()),
            max_l1=float(np.abs(extras['w']).sum(-1).max()),mean_abs_beta=float(np.abs(extras['beta']).mean()),
            mean_memory_count=float(extras['valid_mask'].sum(-1).mean()))
    log.update(block=block,source_sha256=hashes,price_prefix=audit,job_seconds=time.perf_counter()-start)
    jwrite(receipt,log);return log

def build_neural_outputs(out):
    out=Path(out);p,dates,_=core.load_prefix(out/'approved_prefix.csv');origins=np.arange(459,930)
    allpred={};fitlogs=[]
    for mode in core.MODES:
        modes=[]
        for seed in core.SEEDS:
            parts=[]
            for b in range(12):
                stem=out/'fits'/f'{mode}_s{seed}_b{b:02d}'
                with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:parts.append(z['prediction'])
                fitlogs.append(json.loads(stem.with_suffix('.json').read_text()))
            modes.append(np.concatenate(parts))
        allpred[mode]=np.stack(modes)
    out.joinpath('predictions').mkdir(exist_ok=True)
    for k,v in allpred.items():np.savez_compressed(out/'predictions'/f'{k}.npz',prediction=v)
    truth=np.stack([p[t+1:t+21]-p[t] for t in origins]);years=np.array([d[:4] for d in dates[origins]])
    valid=np.array([dates[t][:4]==dates[t+20][:4] for t in origins])
    np.savez_compressed(out/'evaluation.npz',origins=origins,truth=truth,years=years,valid=valid)
    jwrite(out/'FIT_INDEX.json',[{k:v for k,v in a.items() if k!='trace'} for a in fitlogs])
    return allpred

def run(data,out,workers=3):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p,d,receipt=core.load_prefix(data)
    # Keep only authorized bytes. No reading beyond the approved prefix.
    with Path(data).open('rb',buffering=0) as f:raw=f.read(29204)
    (out/'approved_prefix.csv').write_bytes(raw);jwrite(out/'DATA_RECEIPT.json',receipt)
    jobs=[(str(out/'approved_prefix.csv'),str(out),m,s,b) for m in core.MODES for s in core.SEEDS for b in range(12)]
    start=time.perf_counter();completed=0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(worker,j):j for j in jobs}
        for fut in as_completed(futures):
            log=fut.result();completed+=1
            state=dict(stage='NEURAL_FITS',completed=completed,total=len(jobs),last=[log['mode'],log['seed'],log['block']],elapsed_seconds=time.perf_counter()-start)
            jwrite(out/'STATUS.json',state)
            print(f"{completed}/{len(jobs)} {log['mode']} {log['seed']} b{log['block']} epochs={log['epochs']} selected={log['selected_epoch']} sec={log['seconds']:.2f}",flush=True)
    build_neural_outputs(out);jwrite(out/'NEURAL_RUN.json',dict(completed=True,fits=len(jobs),seconds=time.perf_counter()-start,workers=workers,source_sha256=source_hashes()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();run(a.data,a.out,a.workers)

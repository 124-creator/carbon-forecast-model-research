"""Fresh rolling fits; fixed protocol, separate output directories and numeric weights."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,hashlib,json,platform,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from torch import nn
import core

def jwrite(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def fit(p,ref,end,mode,seed):
    core.configure();ids,x,y,tr,va,sd=core.support(p,end)
    m=core.make_model(mode,seed)
    tx,tq,tb=core.batch(p,ref,ids,sd);ty=torch.as_tensor(y/sd)
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
    with torch.no_grad():initial=float((m(tx[va],tq[va],tb[va])-ty[va]).square().mean())
    state=copy.deepcopy(m.state_dict());best=initial;selected=0;wait=0;trace=[]
    for epoch in range(160):
        opt.zero_grad();pred,info=m(tx[tr],tq[tr],tb[tr],details=True)
        raw=(pred-ty[tr]).square().mean()
        penalty=info['correction'].square().mean() if mode!='GRU' else torch.zeros(())
        loss=raw+penalty;loss.backward();grad=float(nn.utils.clip_grad_norm_(m.parameters(),1.))
        opt.step()
        with torch.no_grad():val=float((m(tx[va],tq[va],tb[va])-ty[va]).square().mean())
        row=[epoch+1,float(raw.detach()),float(penalty.detach()),float(loss.detach()),val,grad]
        if not np.isfinite(row).all():raise FloatingPointError('Non-finite training; no recovery tuning')
        trace.append(row)
        if val<best-1e-6:best=val;state=copy.deepcopy(m.state_dict());selected=epoch+1;wait=0
        else:wait+=1
        if wait>=20:break
    m.load_state_dict(state)
    receipt=dict(mode=mode,seed=seed,end=end,train_n=len(tr),validation_n=len(va),
        first_origin=int(ids[0]),max_training_target=int(ids[tr[-1]]+20),
        first_validation_origin=int(ids[va[0]]),max_validation_target=int(ids[-1]+20),
        scale=sd,initial_validation=initial,best_validation=best,selected_epoch=selected,
        epochs=len(trace),parameter_count=sum(v.numel() for v in m.parameters()),trace=trace)
    return m,sd,receipt

def predict(m,sd,p,ref,orig):
    args=core.batch(p,ref,orig,sd)
    with torch.no_grad():pred,info=m(*args,details=True)
    out={k:v.detach().numpy() for k,v in info.items()}
    return pred.numpy()*sd,out

def run_trajectory(args):
    mode,seed,out=args;out=Path(out);core.configure();p,dates,_=core.load_prefix(out/'approved_prefix.csv')
    ref=core.reference(p);ori=np.arange(459,930);prediction=np.full((len(ori),20),np.nan)
    count=0;t0=time.perf_counter()
    for b in range(12):
        end=459+b*40;sel=np.flatnonzero((ori-459)//40==b);start=time.perf_counter()
        m,sd,receipt=fit(p,ref,end,mode,seed);pr,detail=predict(m,sd,p,ref,ori[sel]);prediction[sel]=pr
        receipt.update(block=b,seconds=time.perf_counter()-start,prediction_origin_min=int(ori[sel].min()),prediction_origin_max=int(ori[sel].max()))
        stem=f'{mode}_s{seed}_b{b:02d}'
        values={f'param__{k}':v.detach().numpy() for k,v in m.state_dict().items()}
        values.update(prediction=pr,origins=ori[sel],scale=np.array(sd))
        for k,v in detail.items():values[f'detail__{k}']=v
        np.savez_compressed(out/'fits'/f'{stem}.npz',**values);jwrite(out/'fits'/f'{stem}.json',receipt)
        print(f'{mode} seed={seed} block={b} epochs={receipt["epochs"]} chosen={receipt["selected_epoch"]}',flush=True);count+=1
    if not np.isfinite(prediction).all():raise ValueError('Incomplete trajectory')
    np.savez_compressed(out/'trajectories'/f'{mode}_{seed}.npz',prediction=prediction)
    return dict(mode=mode,seed=seed,fit_count=count,seconds=time.perf_counter()-t0)

def run(out,prefix,workers=3):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a new output directory; no hidden resume/overwrite')
    out.mkdir(parents=True);[ (out/k).mkdir() for k in ('fits','predictions','trajectories') ]
    p,dates,receipt=core.load_prefix(prefix)
    with Path(prefix).open('rb',buffering=0) as f:(out/'approved_prefix.csv').write_bytes(f.read(29204))
    jwrite(out/'DATA_RECEIPT.json',receipt)
    ori=np.arange(459,930);truth=np.stack([p[t+1:t+21]-p[t] for t in ori]);year=dates[ori].astype('U4')
    eligible=dates[ori].astype('U4')==dates[ori+20].astype('U4')
    np.savez_compressed(out/'EVALUATION.npz',origins=ori,truth=truth,years=year,eligible=eligible)
    import scipy,sklearn
    jwrite(out/'ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,workers=workers,dtype='float64',mkldnn=False))
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    jwrite(out/'EXECUTED_SOURCE_HASHES.json',source)
    start=time.perf_counter();records=[]
    jobs=[(mode,s,str(out)) for mode in core.MODES for s in core.SEEDS]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future=[pool.submit(run_trajectory,j) for j in jobs]
            for f in as_completed(future):records.append(f.result());jwrite(out/'TRAJECTORY_REGISTER.json',records)
    except Exception:
        (out/'FAILURE.txt').write_text(traceback.format_exc());raise
    for mode in core.MODES:
        pr=[]
        for s in core.SEEDS:
            with np.load(out/'trajectories'/f'{mode}_{s}.npz',allow_pickle=False) as z:pr.append(z['prediction'].copy())
        np.savez_compressed(out/'predictions'/f'{mode}.npz',prediction=np.stack(pr))
    jwrite(out/'TRAINING_RUN.json',dict(complete=True,formal_fits=sum(r['fit_count'] for r in records),seconds=time.perf_counter()-start,source_hashes=source,training_scope='new models, repeated development, no holdout'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--prefix',required=True);a.add_argument('--workers',type=int,default=3);o=a.parse_args();run(o.out,o.prefix,o.workers)

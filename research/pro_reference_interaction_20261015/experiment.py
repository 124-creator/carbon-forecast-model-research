"""Bounded fresh experiments. Every issued forecast and failed fit is retained."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, copy, hashlib, json, time, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from model import (SEEDS,MODES,LAMBDAS,HORIZONS,load_prefix,reference,
                   state_features,tensor_arrays,InteractionGRU,PlainGRU,predict)

P=None; D=None; REF=None; OUT=None

def dump(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def fit_one(p, ref, end, mode, lam, seed):
    setup(); start=time.perf_counter(); torch.manual_seed(seed)
    model=PlainGRU() if mode=='GRU' else InteractionGRU(mode)
    ids, tensors,tr,va,sd,receipt=tensor_arrays(p,ref,end)
    x,q,b,y=tensors
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    with torch.no_grad():
        pred,_=model(x[va],q[va],b[va]); best=float(((pred-y[va])**2).mean())
    state=copy.deepcopy(model.state_dict()); selected=0; wait=0; history=[]
    for ep in range(160):
        opt.zero_grad(set_to_none=True)
        pred,corr=model(x[tr],q[tr],b[tr])
        mse=((pred-y[tr])**2).mean()
        penalty=lam*(corr*corr).mean() if mode!='GRU' else mse*0
        loss=mse+penalty
        loss.backward(); gn=float(nn.utils.clip_grad_norm_(model.parameters(),1.))
        opt.step()
        with torch.no_grad():
            pv,cv=model(x[va],q[va],b[va]); vl=float(((pv-y[va])**2).mean())
        row=[float(mse.detach()),float(penalty.detach()),vl,gn]
        if not np.isfinite(row).all():
            raise FloatingPointError('Non-finite fit, no silent replacement')
        history.append(row)
        if vl<best-1e-6:
            best=vl;state=copy.deepcopy(model.state_dict());selected=ep+1;wait=0
        else:
            wait+=1
        if wait>=20:
            break
    model.load_state_dict(state)
    receipt.update(mode=mode,lambda_value=lam,seed=seed,selected_epoch=selected,
                   epochs=len(history),best_validation_mse=best,
                   parameters=sum(a.numel() for a in model.parameters()),
                   history_columns=['train_mse','correction_penalty','val_mse','raw_grad_norm'],
                   history=history,seconds=time.perf_counter()-start)
    return model,sd,receipt

def initialize(data_path,out):
    global P,D,REF,OUT
    setup(); P,D,_=load_prefix(data_path);REF=reference(P,240);OUT=Path(out)

def job(task):
    mode,lam,seed,block=task;end=459+40*block
    tag=f'{mode}_l{lam:g}_s{seed}_b{block:02d}'
    model,sd,receipt=fit_one(P,REF,end,mode,lam,seed)
    ids=np.arange(end,min(end+40,930))
    pred=predict(model,sd,P,REF,ids)
    if not np.isfinite(pred).all():raise FloatingPointError('Non-finite issued forecasts')
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    np.savez_compressed(OUT/'fits'/f'{tag}.npz',origins=ids,prediction=pred,scale=np.array(sd),**weights)
    dump(OUT/'fits'/f'{tag}.json',dict(id=tag,block=block,**receipt))
    return tag,receipt['epochs'],receipt['selected_epoch'],receipt['seconds']

def xlags(p,ids,lag,sd):
    return np.array([np.diff(p[t-lag:t+1]) for t in ids])/sd

def baselines(p,dates,refs,out):
    """Full declared pool; trees fit only evaluated horizons, disclosed."""
    allorig=np.arange(459,930)
    arrays={'RW':np.zeros((950,20))}
    for w,ref in refs.items():arrays[f'ML_w{w}']=ref['b'].copy()
    logs=[]
    for w in (120,240,480):
        pr=np.zeros((950,20))
        for t in allorig:pr[t]=np.mean(np.diff(p[max(0,t-w):t+1]))*np.arange(1,21)
        arrays[f'DRIFT_w{w}']=pr
    for block in range(12):
        end=459+40*block
        ids,tensors,tr,va,sd,rec=tensor_arrays(p,refs[240],end)
        ti=ids[tr]; oi=np.arange(end,min(end+40,930))
        yy=np.array([p[t+1:t+21]-p[t] for t in ti])/sd
        for lag in (5,20):
            xx=xlags(p,ti,lag,sd);xp=xlags(p,oi,lag,sd)
            for alpha in (.01,.1,1.,10.):
                tag=f'RIDGE_L{lag}_a{alpha:g}';arrays.setdefault(tag,np.full((950,20),np.nan))
                st=time.perf_counter();m=Ridge(alpha=alpha*len(ti),fit_intercept=True).fit(xx,yy)
                arrays[tag][oi]=m.predict(xp)*sd
                logs.append(dict(config=tag,block=block,seconds=time.perf_counter()-st,**rec))
        xx=xlags(p,ti,20,sd);xp=xlags(p,oi,20,sd)
        for leaves in (7,15):
            for rate in (.05,.1):
                tag=f'TREE_l{leaves}_r{rate:g}';arrays.setdefault(tag,np.full((950,20),np.nan))
                st=time.perf_counter()
                for h in HORIZONS:
                    m=HistGradientBoostingRegressor(max_leaf_nodes=leaves,learning_rate=rate,
                        max_iter=80,min_samples_leaf=20,early_stopping=False,random_state=17)
                    m.fit(xx,yy[:,h-1]);arrays[tag][oi,h-1]=m.predict(xp)*sd
                logs.append(dict(config=tag,block=block,fit_horizons=list(HORIZONS),
                                 seconds=time.perf_counter()-st,**rec))
        qtr=state_features(refs[240],ti,sd);qpr=state_features(refs[240],oi,sd)
        er=yy-refs[240]['b'][ti]/sd
        for alpha in (.01,.1,1.):
            tag=f'LINEAR_RESIDUAL_a{alpha:g}';arrays.setdefault(tag,np.full((950,20),np.nan))
            st=time.perf_counter();m=Ridge(alpha=alpha*len(ti),fit_intercept=True).fit(qtr,er)
            arrays[tag][oi]=refs[240]['b'][oi]+sd*m.predict(qpr)
            logs.append(dict(config=tag,block=block,seconds=time.perf_counter()-st,**rec))
    dump(out/'baseline_fits.json',logs)
    for name,pr in arrays.items():np.savez_compressed(out/'predictions'/f'{name}.npz',pred=pr[None])
    return arrays

def restore_plain(path):
    m=PlainGRU()
    with np.load(path,allow_pickle=False) as z:
        sd=float(z['scale']);state={k:torch.from_numpy(z[k].copy()) for k in m.state_dict()}
    m.load_state_dict(state);return m,sd

def fast_path(p,ref,out,seed):
    """Use newly fitted initial model; each matured (origin,h) consumed once."""
    m,sd=restore_plain(out/'fits'/f'GRU_l0_s{seed}_b00.npz')
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.001)
    pr=np.full((950,20),np.nan);logs=[]
    for t in range(459,930):
        origins=np.arange(max(459,t-20),t)
        if len(origins):
            horizons=t-origins-1
            x=torch.tensor(xlags(p,origins,60,sd),dtype=torch.float32)
            target=torch.tensor((p[t]-p[origins])/sd,dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            delta=m(x)[0][torch.arange(len(origins)),torch.tensor(horizons)]
            loss=((delta-target)**2).mean();loss.backward()
            gn=float(nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
            logs.append(dict(time=t,origin_min=int(origins[0]),origin_max=int(origins[-1]),
                             count=len(origins),loss=float(loss.detach()),gradient_norm=gn))
        pr[t]=predict(m,sd,p,ref,np.array([t]))[0]
        if not np.isfinite(pr[t]).all():raise FloatingPointError('Non-finite online forecast')
    dump(out/'online'/f's{seed}.json',logs)
    np.savez_compressed(out/'online'/f's{seed}_final.npz',**{k:v.detach().numpy() for k,v in m.state_dict().items()})
    return pr

def matured_mix(p,a,b):
    """OLS combination coefficient by horizon on last120 matured issued errors."""
    pr=np.full((950,20),np.nan);weights=np.full((950,20),np.nan)
    for t in range(459,930):
        for h in range(1,21):
            past=np.arange(max(459,t-h-119),t-h+1)
            w=.5
            if len(past)>=30:
                change=a[past,h-1]-b[past,h-1]
                err=p[past+h]-p[past]-b[past,h-1]
                denom=float(change@change)
                w=float(np.clip(change@err/denom,0,1)) if denom>1e-12 else .5
            weights[t,h-1]=w;pr[t,h-1]=w*a[t,h-1]+(1-w)*b[t,h-1]
    return pr,weights

def run(data_path: Path,out: Path,workers: int=3):
    if out.exists():raise FileExistsError('Use a new directory to preserve first results')
    out.mkdir(parents=True)
    for d in ('fits','predictions','online'): (out/d).mkdir()
    p,dates,receipt=load_prefix(data_path)
    refs={w:reference(p,w) for w in (120,240,480)}
    dump(out/'data_receipt.json',receipt)
    code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    dump(out/'source_hashes_before_fits.json',code)
    dump(out/'environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,workers=workers))
    np.savez_compressed(out/'reference_states.npz',**{f'w{w}_{k}':v for w,r in refs.items() for k,v in r.items()})
    tasks=[(m,lam,s,b) for m in MODES for lam in LAMBDAS for s in SEEDS for b in range(12)]
    tasks += [('GRU',0.,s,b) for s in SEEDS for b in range(12)]
    start=time.perf_counter()
    if workers==1:
        initialize(str(data_path),str(out))
        completed=map(job,tasks)
        for i,record in enumerate(completed,1):
            dump(out/'STATUS.json',dict(stage='neural',complete=i,total=len(tasks),last=record[0]))
            print(i,len(tasks),*record,flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(data_path),str(out))) as pool:
            futures=[pool.submit(job,t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                record=f.result()
                dump(out/'STATUS.json',dict(stage='neural',complete=i,total=len(tasks),last=record[0]))
                print(i,len(tasks),*record,flush=True)
    neural_wall=time.perf_counter()-start
    setup();grouped={}
    for m,lam in [(m,l) for m in MODES for l in LAMBDAS]+[('GRU',0.)]:
        pred=np.full((4,950,20),np.nan)
        for si,s in enumerate(SEEDS):
            for block in range(12):
                with np.load(out/'fits'/f'{m}_l{lam:g}_s{s}_b{block:02d}.npz',allow_pickle=False) as z:
                    pred[si,z['origins']]=z['prediction']
        tag=f'{m}_l{lam:g}';grouped[tag]=pred
        np.savez_compressed(out/'predictions'/f'{tag}.npz',pred=pred)
    st=time.perf_counter();baselines(p,dates,refs,out);baseline_wall=time.perf_counter()-st
    st=time.perf_counter();fast=np.stack([fast_path(p,refs[240],out,s) for s in SEEDS]);slow=grouped['GRU_l0']
    np.savez_compressed(out/'predictions/GRU_FAST.npz',pred=fast)
    np.savez_compressed(out/'predictions/PAIR_FIXED.npz',pred=(fast+slow)/2)
    refpred=refs[240]['b']
    np.savez_compressed(out/'predictions/CORE_GRU_FIXED.npz',pred=(slow+refpred[None])/2)
    for name,aa,bb in [('PAIR_MATURE',fast,slow),('CORE_GRU_MATURE',slow,np.broadcast_to(refpred,slow.shape))]:
        pairs=[matured_mix(p,a,b) for a,b in zip(aa,bb)]
        np.savez_compressed(out/'predictions'/f'{name}.npz',pred=np.stack([t[0] for t in pairs]))
        np.savez_compressed(out/'online'/f'{name}_weights.npz',weights=np.stack([t[1] for t in pairs]))
    online_wall=time.perf_counter()-st
    dump(out/'runtime.json',dict(neural_wall_seconds=neural_wall,baseline_wall_seconds=baseline_wall,
         online_wall_seconds=online_wall,neural_fits=len(tasks),total_wall_seconds=time.perf_counter()-start))
    dump(out/'STATUS.json',dict(stage='complete_unscored',neural_fits=len(tasks),later_prices_read=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args()
    run(a.data,a.out,a.workers)

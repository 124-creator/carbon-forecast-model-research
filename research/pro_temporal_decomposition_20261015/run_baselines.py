"""Fresh fixed baseline pool, same permitted price prefix; no old predictions."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from model import read_prefix,training_arrays
from run_market import dump
H=np.array([1,5,20]);ROOT=Path(__file__).resolve().parent

def lowdim_bank(p):
    rho=np.array([0.]+[r for r in (0.,.25,.5,.75,.9,.97) for _ in range(5)])
    q=np.array([1.]+list((.02,.1,.25,.5,.8))*6);r=1-q
    z=np.zeros(31);P=r/(1-rho**2)
    e=np.zeros((len(p),31));S=np.ones_like(e);d=np.zeros((len(p),31,3))
    for t in range(1,len(p)):
        ee=p[t]-p[t-1]-(rho-1)*z;ss=(rho-1)**2*P+q+r;cc=rho*(rho-1)*P+r
        z=rho*z+cc*ee/ss;P=np.maximum(rho**2*P+r-cc**2/ss,0)
        e[t]=ee;S[t]=ss;d[t]=(rho[:,None]**H[None,:]-1)*z[:,None]
    return e,S,d

def baseline_predictions(p,dates):
    origins=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]],int)
    preds={'RW':np.zeros((len(origins),3))};logs=[];dr=np.diff(p)
    for w in (120,240,480):
        preds[f'DRIFT_w{w}']=np.array([H*dr[max(0,t-w):t].mean() for t in origins])
    e,S,d=lowdim_bank(p)
    configs=[('RIDGE',L,a) for L in (5,20) for a in (.001,.01,.1,1.)]
    configs += [('TREE',leaves,rate,w) for leaves in (7,15) for rate in (.05,.1) for w in (240,480)]
    for w in (120,240,480):preds[f'ML_w{w}']=np.full((len(origins),3),np.nan)
    for b in range(12):
        end=459+40*b;at=np.flatnonzero((origins-459)//40==b);o=origins[at]
        ids,x,y,F,C,A,B,pool,sd,receipt=training_arrays(p,end)
        rawtest=np.array([np.diff(p[t-60:t+1]) for t in o]);targets=y[F][:,H-1]
        for w in (120,240,480):
            lo=max(1,end-w);ss=S[lo:end];ee=e[lo:end];n=len(ee)
            ell=np.log(ss).sum(0)+n*np.log(np.maximum(np.mean(ee**2/ss,0),1e-30))
            j=int(np.argmin(ell));preds[f'ML_w{w}'][at]=d[o,j,:]
            logs.append(dict(family='ML',config=f'ML_w{w}',block=b,end_exclusive=end,likelihood_n=n,selected_bank_index=j,bank_size=31,fit_target_last=end-1))
        for conf in configs:
            start=time.perf_counter()
            if conf[0]=='RIDGE':
                _,L,a=conf;name=f'RIDGE_L{L}_a{a}'
                X=x[F,-L:];mu=X.mean(0);sc=X.std(0);sc=np.maximum(sc,1e-6)
                X=(X-mu)/sc;XX=(rawtest[:,-L:]-mu)/sc;ym=targets.mean(0)
                beta=np.linalg.solve(X.T@X/len(X)+a*np.eye(L),X.T@(targets-ym)/len(X))
                pp=XX@beta+ym;nn=len(F);heads=3
            else:
                _,leaves,rate,w=conf;name=f'TREE_l{leaves}_r{rate}_w{w}'
                ix=F[-w:];X=x[ix,-20:];XX=rawtest[:,-20:];pp=np.empty((len(o),3))
                for k,h in enumerate(H):
                    reg=HistGradientBoostingRegressor(max_leaf_nodes=leaves,learning_rate=rate,max_iter=80,early_stopping=False,random_state=0)
                    reg.fit(X,y[ix,h-1]);pp[:,k]=reg.predict(XX)
                nn=len(ix);heads=3
            if name not in preds:preds[name]=np.full((len(origins),3),np.nan)
            preds[name][at]=pp
            logs.append(dict(family=conf[0],config=name,block=b,end_exclusive=end,n_training_origins=nn,heads=heads,max_train_target=receipt['F_last_target'],seconds=time.perf_counter()-start))
        print('baseline block',b,'complete',flush=True)
    if len(preds)!=23 or any(not np.isfinite(v).all() for v in preds.values()):raise ValueError('baseline coverage')
    return preds,logs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'data/guangzhou_prefix950.csv'));ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'BASELINE_PREDICTIONS.npz').exists():raise FileExistsError('do not overwrite baseline predictions')
    p,dates,audit=read_prefix(a.data);start=time.perf_counter();pred,log=baseline_predictions(p,dates)
    np.savez_compressed(out/'BASELINE_PREDICTIONS.npz',**pred);dump(out/'BASELINE_FIT_LOGS.json',log)
    dump(out/'BASELINE_RECEIPT.json',dict(configurations=len(pred),fit_records=len(log),seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data=audit,price_information='available prefix only',direct_support='F excluding C and purge',ML_support='all causal filter states, trailing one-step likelihood; not identical labeled rows to direct models',tree_targets='1/5/20 only, shared fully matured origin support'))
if __name__=='__main__':main()

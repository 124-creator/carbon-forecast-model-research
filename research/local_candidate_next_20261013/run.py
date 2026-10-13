"""Bounded diagnostic, not a novel forecasting candidate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'deep_direction_review_20261013'/'calibration'))
import calibrate as base
base.OUT=OUT

def features(p,dates,bank,group):
    n=len(p); run=np.zeros(n); gap=np.ones(n)
    dd=np.asarray(dates,dtype='datetime64[D]')
    gap[1:]=(dd[1:]-dd[:-1]).astype(int)
    for t in range(1,n): run[t]=run[t-1]+1 if p[t]==p[t-1] else 0
    quote=np.column_stack([np.log1p(run),np.log1p(gap)])
    trend=np.zeros((n,2))
    for t in range(n):
        for j,w in enumerate([20,60]):
            s=max(0,t-w);trend[t,j]=(p[t]-p[s])/max(t-s,1)
    if group=='intercept':return np.zeros((n,0))
    q=np.column_stack([quote,quote[:,0,None]*bank,quote[:,1,None]*bank])
    if group=='quote':return q
    if group=='trend':return trend
    if group=='all':return np.column_stack([trend,q])
    raise ValueError(group)

def probe(p,dates,fit_end,origins,group,alpha):
    # Never compute even filter states beyond the latest requested origin.
    end=int(max(origins))+1
    bank,ll,pars=base.filter_bank(p[:end])
    start=max(1,fit_end-480);n=fit_end-start;a=ll[start:fit_end].sum(0)
    j=int(np.argmin(a[:,0]+n*np.log(np.maximum(a[:,1]/n,1e-14))))
    ref=bank[:,j];X=features(p[:end],dates[:end],ref,group)
    idx=np.arange(max(60,fit_end-20-480),fit_end-20)
    if idx[-1]+20>=fit_end:raise ValueError('Maturity violation')
    Y=p[idx[:,None]+base.H]-p[idx,None]-ref[idx]
    mu=X[idx].mean(0);sd=X[idx].std(0);sd[sd==0]=1
    A=(X[idx]-mu)/sd;ym=Y.mean(0)
    B=np.linalg.solve(A.T@A+alpha*len(idx)*np.eye(A.shape[1]),A.T@(Y-ym))
    pred=ref[origins]+((X[origins]-mu)/sd)@B+ym
    return pred,dict(fit_end=int(fit_end),max_parameter_price_index=int(fit_end-1),max_training_target=int(idx[-1]+20),n_training_origins=len(idx),rho=float(pars[j,0]),permanent_fraction=float(pars[j,1]),coefficients=B.tolist(),residual_intercept=ym.tolist())

def main():
    begin=time.monotonic();dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'))
    origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
    if len(origins)!=451:raise ValueError('451 protocol failed')
    true=p[origins[:,None]+base.H]-p[origins,None];years=np.array([d[:4] for d in dates[origins]])
    cache=dict(raw=base.raw_features(p),bank=base.filter_bank(p));results=[];logs=[]
    saved=dict(origins=origins,dates=dates[origins],true_changes=true)
    configs=base.configurations()+[dict(family='PROBE',group=g,alpha=a) for g in ['intercept','trend','quote','all'] for a in [1.,10.]]
    base.dump('CONFIGURATIONS.json',configs)
    for c in configs:
        name=c['family']+'_'+ '_'.join(f'{k}{v}' for k,v in c.items() if k!='family');pr=np.full(true.shape,np.nan)
        for fit in range(initial,len(p),40):
            pos=np.where((origins>=fit)&(origins<fit+40))[0]
            if not len(pos):continue
            if c['family']=='PROBE':a,log=probe(p,dates,fit,origins[pos],c['group'],c['alpha'])
            else:a,log=base.predict(p,fit,origins[pos],c,cache)
            pr[pos]=a;log.update(name=name);logs.append(log)
        if not np.isfinite(pr).all():raise ValueError(name)
        score,rows=base.score(pr,true,years)
        results.append(dict(name=name,configuration=c,score=score,by_year=rows));saved[name]=pr
        print(name,score,flush=True)
    mle=saved['MLE_window480'];diag=[]
    f=features(p,dates,np.zeros((len(p),3)),'quote');tr=features(p,dates,np.zeros((len(p),3)),'trend')
    groups={'same_price':f[origins,0]>0,'changed_price':f[origins,0]==0,'gap_gt1':f[origins,1]>np.log(2)+1e-12,'gap_1':f[origins,1]<=np.log(2)+1e-12,'trend20_nonnegative':tr[origins,0]>=0,'trend20_negative':tr[origins,0]<0}
    for y in ['2020','2021']:
        for g,mask in groups.items():
            m=mask&(years==y)
            if not m.any():continue
            err=mle[m]-true[m]
            diag.append(dict(year=y,group=g,count=int(m.sum()),mean_error_h=err.mean(0).tolist(),mse_h=(err**2).mean(0).tolist(),rw_mse_h=(true[m]**2).mean(0).tolist()))
    rng=np.random.default_rng(20261013);boots={r['name']:[] for r in results if r['configuration']['family']=='PROBE'}
    for _ in range(500):
        cells={k:[] for k in boots}
        for y in ['2020','2021']:
            ix=np.where(years==y)[0];starts=rng.integers(0,len(ix),size=int(np.ceil(len(ix)/20)))
            take=ix[np.concatenate([(s+np.arange(20))%len(ix) for s in starts])[:len(ix)]]
            den=np.mean(true[take]**2,0)
            for k in boots:cells[k].extend((((saved[k][take]-true[take])**2-(mle[take]-true[take])**2).mean(0)/den).tolist())
        for k in boots:boots[k].append(float(np.mean(cells[k])))
    ci={k:dict(seed=20261013,reps=500,block=20,quantiles=np.quantile(v,[.025,.5,.975]).tolist()) for k,v in boots.items()}
    np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
    base.dump('ALL_RESULTS.json',results);base.dump('TRAINING_LOGS.json',logs);base.dump('DIAGNOSTICS.json',diag);base.dump('BLOCK_UNCERTAINTY.json',ci)
    base.dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-begin,budget_seconds=1200,configs=len(configs),fit_requests=len(logs),origins=len(origins),initial=initial,candidate_count=0,development_only=True))

if __name__=='__main__':main()

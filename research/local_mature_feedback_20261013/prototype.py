import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,time,hashlib,sys,platform
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
H=np.array([1,5,20])
METHODS=['mature_mtl','complete_mtl','identity_transport','learned_transport']
START=time.monotonic()

def support(t,h,window):
    return np.arange(max(20,t-window),t-h+1,dtype=int)

def grad(A,B,X,y,k,ids):
    z=np.tanh(X[ids]@A)
    err=z@B[:,k]-y[ids,k]
    gb=z.T@err/len(ids)
    ga=X[ids].T@((err[:,None]*B[:,k])* (1-z*z))/len(ids)
    return ga,gb

def model_predict(A,B,x): return np.tanh(x@A)@B

def train(X,y,dates,first,seed,method):
    rng=np.random.default_rng(seed)
    A=rng.normal(0,.12,(X.shape[1],6)); B=rng.normal(0,.08,(6,3))
    numerator=np.zeros((2,*A.shape)); denominator=np.zeros_like(numerator)
    pred=np.full_like(y,np.nan); trace=[]; max_label=-1
    # Chronological replay, never targets not yet matured.
    for t in range(first,len(X)-20):
        if time.monotonic()-START>1800: raise TimeoutError('30 minute computation budget reached')
        for step in range(2):
            q=rng.random(32)
            common=support(t,20,240)
            ci=common[np.minimum((q*len(common)).astype(int),len(common)-1)]
            gs=[]; bs=[]
            for k,h in enumerate(H):
                pool=common if method=='complete_mtl' else support(t,int(h),240)
                ids=pool[np.minimum((q*len(pool)).astype(int),len(pool)-1)]
                if np.max(ids+h)>t: raise RuntimeError('unmatured target accessed')
                max_label=max(max_label,int(np.max(ids+h)))
                ga,gb=grad(A,B,X,y,k,ids); gs.append(ga); bs.append(gb)
            total=sum(gs)/3
            if method in ['identity_transport','learned_transport']:
                # Fully matured age cohorts: 20 origins versus preceding 20.
                recent=np.arange(t-39,t-19); old=np.arange(t-59,t-39)
                if recent.min()<20: raise RuntimeError('insufficient mature anchor')
                delta=[]
                for k in range(3):
                    delta.append(grad(A,B,X,y,k,recent)[0]-grad(A,B,X,y,k,old)[0])
                for k in range(2):
                    numerator[k]=.95*numerator[k]+.05*delta[k]*delta[2]
                    denominator[k]=.95*denominator[k]+.05*delta[k]**2
                    transfer=np.ones_like(A) if method=='identity_transport' else np.clip(numerator[k]/(denominator[k]+.001),-2,2)
                    anchor=grad(A,B,X,y,k,ci)[0]
                    total+=.25*transfer*(gs[k]-anchor)/3
            gB=np.stack(bs,axis=1)/3
            norm=np.sqrt(np.sum(total**2)+np.sum(gB**2)); scale=min(1,10/max(norm,1e-12))
            A-=.01*scale*(total+.001*A); B-=.01*scale*(gB+.001*B)
        pred[t]=model_predict(A,B,X[t])
        if not np.isfinite(pred[t]).all(): raise RuntimeError('nonfinite prediction')
        if t%100==0: trace.append({'t':t,'max_mature_endpoint':max_label,'representation_norm':float(np.linalg.norm(A))})
    return pred,trace

def load(path):
    ds=[]; ps=[]; boundary=None
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            d=row['date']
            if d>'2021-12-31':
                boundary=d; break # no parsing or retaining future prices
            ds.append(d);ps.append(float(row['author_reported_price']))
    if ds!=sorted(ds) or len(set(ds))!=len(ds): raise ValueError('dates unsorted/duplicate')
    return np.array(ds),np.array(ps),{'file':str(path),'included_rows':len(ds),'last_included':ds[-1],'boundary_date_only':boundary}

def main():
    rows=[];audit=[];failure=[];traces=[]
    for path in sorted((ROOT/'data/parsed/figshare_28682348').glob('*.csv')):
        dates,p,a=load(path);audit.append(a)
        first=max(300,int(np.searchsorted(dates,'2019-01-01')))
        scale=float(np.std(np.diff(p[:first]),ddof=1))
        if scale<=0: raise ValueError('zero initial scale')
        r=np.r_[0,np.diff(p)/scale]
        X=np.zeros((len(p),21));X[:,-1]=1
        for lag in range(20): X[lag:,lag]=r[:len(r)-lag]
        sd=np.std(X[20:first,:20],axis=0);mu=np.mean(X[20:first,:20],axis=0)
        X[:,:20]=np.clip((X[:,:20]-mu)/np.maximum(sd,.1),-10,10)
        y=np.zeros((len(p),3))
        for k,h in enumerate(H):y[:-h,k]=(p[h:]-p[:-h])/(scale*np.sqrt(h))
        valid=np.arange(len(p)-20);valid=valid[(dates[valid]>='2020-01-01')&(valid>=first)]
        a.update({'initial_end_exclusive':str(dates[first]),'scale':scale,'first_origin':first,'scoring_origins':len(valid)})
        for seed in range(610,615):
            for method in METHODS:
                try:
                    pred,tr=train(X,y,dates,first,seed,method)
                    np.savez_compressed(HERE/f'pred_{path.stem}_{seed}_{method}.npz',origin=valid,dates=dates[valid],prediction=pred[valid],target=y[valid])
                    traces.append({'market':path.stem,'seed':seed,'method':method,'trace':tr})
                    for year in ['2020','2021']:
                        ix=valid[np.char.startswith(dates[valid],year)]
                        for k,h in enumerate(H):
                            loss=float(np.mean((pred[ix,k]-y[ix,k])**2));rw=float(np.mean(y[ix,k]**2))
                            rows.append({'market':path.stem,'seed':seed,'method':method,'year':year,'horizon':int(h),'n':len(ix),'mse':loss,'rw_mse':rw,'ratio':loss/rw})
                    print(path.stem,seed,method,'completed',round(time.monotonic()-START,2),flush=True)
                except Exception as e:
                    failure.append({'market':path.stem,'seed':seed,'method':method,'error':repr(e)});raise
                finally:
                    (HERE/'ALL_RESULTS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                    (HERE/'FAILURES.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
    summary={m:{'mean_seed_cell_ratio':float(np.mean([v['ratio'] for v in rows if v['method']==m])),'mean_seed_cell_mse':float(np.mean([v['mse'] for v in rows if v['method']==m])),'cells':len([v for v in rows if v['method']==m])} for m in METHODS}
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (HERE/'INPUT_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (HERE/'TRAINING_TRACES.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    receipt={'elapsed_seconds':time.monotonic()-START,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'models':len(traces),'result_cells':len(rows),'seeds':list(range(610,615)),'future_prices_parsed':False,'no_pickle_or_old_weights':True}
    (HERE/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

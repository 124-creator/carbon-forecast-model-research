"""One bounded maturity-support x refit-rate diagnostic; no new model."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
H=np.array([1,5,20])

def support(end,h,window,mode):
    if mode not in ['common','per_h']:raise ValueError(mode)
    return np.arange(max(60,end-20-window),end-(20 if mode=='common' else h),dtype=int)

def ridge_predict(p,x,end,origins,mode,window=480,alpha=1.):
    common=support(end,20,window,'common')
    mu=x[common].mean(0);sd=x[common].std(0);sd[sd==0]=1
    out=[];trace=[]
    for h in H:
        ids=support(end,int(h),window,mode)
        if not len(ids) or ids[-1]+h>=end:raise ValueError('unmatured support')
        a=(x[ids]-mu)/sd;y=p[ids+h]-p[ids]
        am=a.mean(0);ym=y.mean();ac=a-am;yc=y-ym
        g=ac.T@ac+alpha*len(ids)*np.eye(x.shape[1]);rhs=ac.T@yc
        b=solve(g,rhs,assume_a='pos')
        out.append((((x[origins]-mu)/sd)-am)@b+ym)
        trace.append(dict(h=int(h),n=len(ids),fit_end_exclusive=end,min_origin=int(ids[0]),max_label_endpoint=int(ids[-1]+h),normal_equation_relative_residual=float(np.linalg.norm(g@b-rhs)/max(1.,np.linalg.norm(rhs)))))
    return np.stack(out,axis=1),trace

def raw_features(p):
    x=np.zeros((len(p),60))
    for t in range(60,len(p)):x[t]=np.diff(p[t-60:t+1])[::-1]
    return x

def load():
    path=ROOT/'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with path.open('rb',buffering=0) as f:
        lines=[f.readline() for _ in range(951)];end=f.tell()
    rows=list(csv.DictReader(b''.join(lines).decode('utf-8-sig').splitlines()))
    d=np.array([r['date'] for r in rows]);p=np.array([float(r['author_reported_price']) for r in rows])
    if len(rows)!=950 or d[-1]!='2021-12-31' or np.any(d[1:]<=d[:-1]) or not np.isfinite(p).all():raise ValueError('cutoff failure')
    return d,p,dict(rows=950,last_date=str(d[-1]),bytes_consumed=end,prefix_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),boundary_requested=False,dependency='Previously independently verified 950-row record count')

def main():
    start=time.monotonic();d,p,audit=load();initial=int(np.sum(d<'2020-01-01'))
    origins=np.array([t for t in range(initial,len(p)-20) if d[t][:4]==d[t+20][:4]])
    if initial!=459 or len(origins)!=451:raise ValueError('unified protocol mismatch')
    y=p[origins[:,None]+H]-p[origins,None];x=raw_features(p)
    saved=dict(origins=origins,dates=d[origins],target=y);results=[];traces=[];checks=0;fit_requests=0
    for mode in ['common','per_h']:
        for refresh in [40,1]:
            name=f'{mode}_refresh{refresh}';pred=np.full(y.shape,np.nan)
            for end in range(initial,len(p),refresh):
                if time.monotonic()-start>900:raise TimeoutError('900 second budget')
                pos=np.where((origins>=end)&(origins<end+refresh))[0]
                if not len(pos):continue
                pr,trace=ridge_predict(p,x,end,origins[pos],mode)
                pred[pos]=pr;fit_requests+=1
                traces.extend([dict(configuration=name,**r) for r in trace])
                # Parameters cannot see current or future boundary prices.
                q=p.copy();q[end:]+=100000.
                qp,_=ridge_predict(q,x,end,origins[pos],mode)
                if not np.array_equal(pr,qp):raise ValueError('future label sensitivity')
                # Current feature is exactly reproducible from available prefix.
                t=int(origins[pos][-1])
                if not np.array_equal(x[t],raw_features(p[:t+1])[-1]):raise ValueError('feature leakage')
                checks+=1
            if not np.isfinite(pred).all():raise ValueError('incomplete forecast')
            cells=[]
            for year in ['2020','2021']:
                mask=np.char.startswith(d[origins],year)
                for k,h in enumerate(H):
                    mse=float(np.mean((pred[mask,k]-y[mask,k])**2));rw=float(np.mean(y[mask,k]**2))
                    cells.append(dict(year=year,h=int(h),n=int(mask.sum()),original_price_mse=mse,rw_mse=rw,ratio=mse/rw))
            results.append(dict(name=name,score=float(np.mean([c['ratio'] for c in cells])),cells=cells));saved[name]=pred
    np.savez_compressed(HERE/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
    npz=np.load(HERE/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
    errors=[]
    for result in results:
        for cell in result['cells']:
            mask=np.char.startswith(npz['dates'],cell['year']);k=list(H).index(cell['h'])
            mse=float(np.mean((npz[result['name']][mask,k]-npz['target'][mask,k])**2))
            errors.append(abs(mse-cell['original_price_mse']))
    receipt=dict(seconds=time.monotonic()-start,budget_seconds=900,fit_requests=fit_requests,scalar_head_solves=fit_requests*3,verification_refits=checks,origins=len(origins),initial_rows=initial,max_normal_residual=max(t['normal_equation_relative_residual'] for t in traces),saved_score_recalculation_error=max(errors),market_data_cutoff='2021-12-31',new_model_count=0)
    if receipt['max_normal_residual']>1e-8 or max(errors)>1e-10:raise ValueError('verification failed')
    for name,obj in [('INPUT_AUDIT.json',audit),('RESULTS.json',results),('TRAINING_TRACES.json',traces),('RECEIPT.json',receipt)]:
        (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(results=[dict(name=r['name'],score=r['score']) for r in results],receipt=receipt),indent=2))
if __name__=='__main__':main()

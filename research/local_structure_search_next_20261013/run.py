import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,time,hashlib,sys
from pathlib import Path
import numpy as np
from prototype import fit
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
H=np.array([1,5,20]);METHODS=['corrected','rank1','diagonal','full','rw'];WINDOWS=[240,480]
START=time.monotonic()
def dump(name,obj): (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def budget():
    if time.monotonic()-START>1800:raise TimeoutError('1800 second budget')
def load():
    path=ROOT/'data/parsed/figshare_28682348/guangzhou_reported.csv'
    lines=[]
    # Unbuffered raw byte stream; do not request any byte after the known 950th record.
    with path.open('rb',buffering=0) as f:
        for _ in range(951):
            line=f.readline()
            if not line:raise ValueError('short CSV prefix')
            lines.append(line)
        end=f.tell()
    rows=list(csv.DictReader(b''.join(lines).decode('utf-8-sig').splitlines()))
    dates=np.array([r['date'] for r in rows]);p=np.array([float(r['author_reported_price']) for r in rows])
    if len(rows)!=950 or dates[-1]!='2021-12-31' or np.any(dates>'2021-12-31'):raise ValueError('invalid cutoff prefix')
    if list(dates)!=sorted(set(dates)) or not np.isfinite(p).all():raise ValueError('invalid input')
    dump('INPUT_AUDIT.json',{'file':str(path),'included_rows':950,'first':str(dates[0]),'last':str(dates[-1]),'bytes_read':end,'read_buffering':0,'prefix_sha256':hashlib.sha256(b''.join(lines)).hexdigest(),'boundary_row_read':False})
    return dates,p

def mature_fit(X,p,t,window,method):
    ids=np.arange(max(20,t-window),t-20+1)
    if np.max(ids+20)>t:raise RuntimeError('unmatured target')
    Y=np.stack([p[ids+h]-p[ids] for h in H],axis=1)
    B,info=fit(X[ids],Y,method,20)
    info.update({'n':len(ids),'max_endpoint':int(max(ids)+20)})
    return X[t]@B,info

def synthetic():
    rows=[]
    for hetero in [False,True]:
        for seed in range(30):
            budget();rng=np.random.default_rng(seed+910)
            X=rng.normal(size=(2120,12));a=rng.normal(size=12);a/=np.linalg.norm(a)
            v=np.array([1.,.6,.3]);v/=np.linalg.norm(v);B=np.outer(a,v)*.7
            scale=np.where(X[:,0]>1,5.,.5) if hetero else np.full(len(X),1.5)
            noise=rng.normal(size=(len(X),3))@np.array([[.5,0,0],[0,.8,0],[0,.6,2.]]).T
            Y=X@B+noise*scale[:,None]
            saved={ 'X_train':X[:120],'Y_train':Y[:120],'X_eval':X[120:],'Y_eval':Y[120:],'true_B':B}
            for method in METHODS:
                b,info=fit(X[:120],Y[:120],method,0)
                pred=X[120:]@b;saved[method]=pred
                _,_,vh=np.linalg.svd(b,full_matrices=False)
                angle=float(1-(vh[0]@v)**2) if method!='rw' else None
                rows.append({'hetero':hetero,'seed':seed,'method':method,'mse':float(np.mean((Y[120:]-pred)**2)),'direction_error':angle,**info})
            np.savez_compressed(HERE/f'synthetic_{int(hetero)}_{seed}.npz',**saved)
    dump('SYNTHETIC_RESULTS.json',rows)
    return {str(h):{m:{'mse':float(np.mean([r['mse'] for r in rows if r['hetero']==h and r['method']==m])),'direction_error':None if m=='rw' else float(np.mean([r['direction_error'] for r in rows if r['hetero']==h and r['method']==m]))} for m in METHODS} for h in [False,True]}

def market():
    dates,p=load();scale=float(np.std(np.diff(p[:300]),ddof=1));r=np.r_[0,np.diff(p)/scale]
    X=np.ones((len(p),21))
    for lag in range(20):X[lag:,lag]=r[:len(r)-lag];X[:lag,lag]=0
    origin=np.arange(300,len(p)-20);origin=origin[dates[origin]>='2020-01-01']
    target=np.stack([p[origin+h]-p[origin] for h in H],axis=1)
    rows=[];traces=[]
    for window in WINDOWS:
        for method in METHODS:
            pred=np.full((len(p),3),np.nan)
            for t in range(300,len(p)-20):
                budget();pred[t],info=mature_fit(X,p,t,window,method)
                traces.append({'window':window,'method':method,'origin':t,**info})
            np.savez_compressed(HERE/f'market_{window}_{method}.npz',origin=origin,dates=dates[origin],prediction=pred[origin],target=target)
            for year in ['2020','2021']:
                mask=np.char.startswith(dates[origin],year)
                for k,h in enumerate(H):
                    mse=float(np.mean((pred[origin[mask],k]-target[mask,k])**2));rw=float(np.mean(target[mask,k]**2))
                    rows.append({'window':window,'method':method,'year':year,'horizon':int(h),'n':int(mask.sum()),'original_price_mse':mse,'rw_mse':rw,'ratio':mse/rw})
            print(window,method,'complete',flush=True)
            dump('MARKET_RESULTS.json',rows);dump('TRAINING_TRACES.json',traces)
    return {str(w):{m:float(np.mean([z['ratio'] for z in rows if z['window']==w and z['method']==m])) for m in METHODS} for w in WINDOWS}

def main():
    dump('CONFIG.json',{'methods':METHODS,'windows':WINDOWS,'horizons':H.tolist(),'initial_scale_end_exclusive':300,'feature_lags':20,'rank':1,'hac_lags_market':20,'synthetic_seeds':list(range(910,940)),'objective':'unweighted original price squared loss','no_horizon_scaling':True,'selection':'none; both prespecified windows reported'})
    try:
        s=synthetic();m=market();dump('SUMMARY.json',{'synthetic':s,'market':m});dump('EXECUTION.json',{'elapsed_seconds':time.monotonic()-START,'numpy':np.__version__,'python':sys.version,'market_training_fits':6300,'synthetic_fits':300,'code_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['run.py','prototype.py']}})
        print(json.dumps({'synthetic':s,'market':m},indent=2))
    except Exception as e:dump('FAILURE.json',{'error':repr(e),'elapsed_seconds':time.monotonic()-START});raise
if __name__=='__main__':main()

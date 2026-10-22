"""Fresh statistical, supervised and simple combination controls; no old weights."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import core
from experiment import jwrite

def run(out):
    out=Path(out);p,dates,audit=core.load_prefix(out/'approved_prefix.csv');ori=np.arange(459,930);dest=out/'predictions';dest.mkdir(exist_ok=True)
    start=time.perf_counter();logs=[];preds={'RW':np.zeros((1,len(ori),20))}
    for w in (120,240,480):
        ref=core.reference(p,w);preds[f'ML_w{w}']=ref['b'][ori][None]
        preds[f'DRIFT_w{w}']=np.stack([np.arange(1,21)*np.diff(p[max(0,t-w):t+1]).mean() for t in ori])[None]
    configs=[('RIDGE',l,a) for l in (5,20) for a in (.01,.1,1.,10.)]+[('TREE',l,a) for l in (7,15) for a in (.05,.1)]
    for fam,l,a in configs:
        name=f'{fam}_{l}_{a}';pr=np.full((1,len(ori),20),np.nan)
        for b in range(12):
            end=459+40*b;ids,x,y,tr,va,sd=core.support(p,end);sel=np.where((ori-459)//40==b)[0]
            xx=np.stack([np.diff(p[t-60:t+1]) for t in ori[sel]])/sd;xt=x[tr]/sd
            if fam=='RIDGE':xt=xt[:,-l:];xx=xx[:,-l:]
            t0=time.perf_counter()
            for h in [1,5,20]:
                model=Ridge(alpha=a,fit_intercept=True) if fam=='RIDGE' else HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=l,learning_rate=a,early_stopping=False,random_state=0)
                model.fit(xt,y[tr,h-1]/sd);pr[0,sel,h-1]=model.predict(xx)*sd
            logs.append(dict(name=name,block=b,end=end,max_training_target=int(ids[tr[-1]]+20),train_n=len(tr),seconds=time.perf_counter()-t0,scored_horizons=[1,5,20]))
        preds[name]=pr
    for name,pr in preds.items():np.savez_compressed(dest/f'{name}.npz',prediction=pr)
    with np.load(dest/'GRU.npz',allow_pickle=False) as z:g=z['prediction']
    base=core.reference(p,240)['b'][ori];truth=np.stack([p[t+1:t+21]-p[t] for t in ori])
    fixed=(g+base[None])/2;weights=np.ones_like(g)*.5
    for s in range(g.shape[0]):
        for t,i in enumerate(ori):
            for h in range(1,21):
                past=np.where(ori+h<=i)[0][-120:]
                if len(past)<30:continue
                d=g[s,past,h-1]-base[past,h-1];e=truth[past,h-1]-base[past,h-1];den=float(d@d)
                weights[s,t,h-1]=np.clip(float(d@e)/den,0.,1.) if den>1e-12 else .5
    mature=base[None]+weights*(g-base[None])
    np.savez_compressed(dest/'CORE_GRU_FIXED.npz',prediction=fixed)
    np.savez_compressed(dest/'CORE_GRU_MATURE.npz',prediction=mature)
    np.savez_compressed(out/'COMBINATION_WEIGHTS.npz',mature=weights)
    np.savez_compressed(out/'REFERENCE.npz',**core.reference(p,240))
    jwrite(out/'BASELINE_FITS.json',logs)
    jwrite(out/'BASELINE_RUN.json',dict(completed=True,base_configurations=19,combination_configurations=2,supervised_refresh_records=len(logs),horizon_fits=len(logs)*3,seconds=time.perf_counter()-start,unfitted_horizons='NaN, not forecasts',data_receipt=audit))
    print('Traditional and combination controls complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)

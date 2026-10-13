import json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
from run import HERE,METHODS,WINDOWS,load,mature_fit,H
from prototype import fit

def main():
    rows=json.loads((HERE/'MARKET_RESULTS.json').read_text(encoding='utf-8'));srows=json.loads((HERE/'SYNTHETIC_RESULTS.json').read_text(encoding='utf-8'))
    traces=json.loads((HERE/'TRAINING_TRACES.json').read_text(encoding='utf-8'))
    if len(rows)!=60 or len(srows)!=300 or len(traces)!=6300:raise ValueError('unexpected count')
    err=0.;count=0
    for w in WINDOWS:
        for method in METHODS:
            with np.load(HERE/f'market_{w}_{method}.npz',allow_pickle=False) as a:
                if max(a['dates'])>'2021-12-31' or len(a['origin'])!=471:raise ValueError('date/origin violation')
                if not np.isfinite(a['prediction']).all():raise ValueError('nonfinite')
                for row in [r for r in rows if r['window']==w and r['method']==method]:
                    ix=np.char.startswith(a['dates'],row['year']);k=list(H).index(row['horizon']);mse=float(np.mean((a['prediction'][ix,k]-a['target'][ix,k])**2));rw=float(np.mean(a['target'][ix,k]**2))
                    err=max(err,abs(mse-row['original_price_mse']),abs(mse/rw-row['ratio']));count+=1
    serr=0
    for hetero in [False,True]:
        for seed in range(30):
            with np.load(HERE/f'synthetic_{int(hetero)}_{seed}.npz',allow_pickle=False) as a:
                for method in METHODS:
                    row=next(r for r in srows if r['hetero']==hetero and r['seed']==seed and r['method']==method)
                    loss=float(np.mean((a['Y_eval']-a[method])**2));serr=max(serr,abs(loss-row['mse']))
                    b,info=fit(a['X_train'],a['Y_train'],method,0)
                    if not np.array_equal(a['X_eval']@b,a[method]):raise ValueError('synthetic replay differs')
    dates,p=load();r=np.r_[0,np.diff(p)/float(np.std(np.diff(p[:300]),ddof=1))];X=np.ones((len(p),21))
    for lag in range(20):X[lag:,lag]=r[:len(r)-lag];X[:lag,lag]=0
    replay=0
    for w in WINDOWS:
        for method in METHODS:
            with np.load(HERE/f'market_{w}_{method}.npz',allow_pickle=False) as a:
                for j in [0,100,470]:
                    t=int(a['origin'][j]);pred,_=mature_fit(X,p,t,w,method)
                    replay=max(replay,float(np.max(abs(pred-a['prediction'][j]))))
                    changed=p.copy();changed[t+1:]+=1e7
                    other,_=mature_fit(X,changed,t,w,method)
                    if not np.array_equal(pred,other):raise ValueError('future label leakage')
    if err>1e-12 or serr>1e-12 or replay>1e-12:raise ValueError('metric/replay mismatch')
    if any(r['max_endpoint']>r['origin'] for r in traces):raise ValueError('maturity violation')
    results={'market_score_cells':count,'synthetic_score_cells':len(srows),'npz_files':70,'market_fitting_requests':len(traces),'actual_non_rw_market_fits':sum(t['method']!='rw' for t in traces),'max_market_metric_error':err,'max_synthetic_metric_error':serr,'sampled_market_replay_error':replay,'synthetic_retraining_predictions_exact':True,'future_perturbation_checks':30,'maturity_all_traces':True,'independent_retraining':'all 300 synthetic requests and 30 sampled market requests; not full market replay','optimized_python':not __debug__}
    (HERE/('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json')).write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
if __name__=='__main__':main()

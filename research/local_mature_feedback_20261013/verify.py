import json,hashlib
from pathlib import Path
import numpy as np
from prototype import grad,train,HERE,METHODS

rows=json.loads((HERE/'ALL_RESULTS.json').read_text())
if len(rows)!=360: raise RuntimeError('wrong cell count')
errors=[];count=0
for path in HERE.glob('pred_*.npz'):
    with np.load(path,allow_pickle=False) as z:
        origin=z['origin'];dates=z['dates'];pred=z['prediction'];target=z['target']
        if not np.isfinite(pred).all() or max(dates.tolist())>'2021-12-31':raise RuntimeError('invalid predictions')
        matched=[r for r in rows if path.stem==f"pred_{r['market']}_{r['seed']}_{r['method']}"]
        if len(matched)!=6:raise RuntimeError('missing result')
        for r in matched:
            mask=np.char.startswith(dates,r['year']); k=[1,5,20].index(r['horizon'])
            mse=float(np.mean((pred[mask,k]-target[mask,k])**2))
            errors.append(abs(mse-r['mse']))
        count+=1
if count!=60:raise RuntimeError('wrong predictions count')
rng=np.random.default_rng(2);X=rng.normal(size=(120,4));y=rng.normal(size=(120,3));A=rng.normal(size=(4,6))*.1;B=rng.normal(size=(6,3))*.1
ids=np.arange(20,60);ga,gb=grad(A,B,X,y,1,ids);eps=1e-6;delta=np.zeros_like(A);delta[1,2]=eps
loss=lambda a:np.mean((np.tanh(X[ids]@a)@B[:,1]-y[ids,1])**2)/2
finite=(loss(A+delta)-loss(A-delta))/(2*eps)
if abs(finite-ga[1,2])>1e-7:raise RuntimeError('gradient mismatch')
# Alter exclusively not-yet-mature labels; current forecast at t=80 must not change.
y2=y.copy();y2[81:,0]+=10000;y2[76:,1]+=10000;y2[61:,2]+=10000
causal_errors={}
for m in METHODS:
    p1,_=train(X,y,np.array(['2020']*120),80,610,m)
    p2,_=train(X,y2,np.array(['2020']*120),80,610,m)
    causal_errors[m]=float(np.max(np.abs(p1[80]-p2[80])))
if max(causal_errors.values())!=0:raise RuntimeError('causality failed')
summary={m:{market:float(np.mean([r['ratio'] for r in rows if r['method']==m and r['market']==market])) for market in sorted(set(r['market'] for r in rows))} for m in METHODS}
paired=[]
for seed in range(610,615):
    candidate=np.mean([r['ratio'] for r in rows if r['seed']==seed and r['method']=='learned_transport'])
    baseline=np.mean([r['ratio'] for r in rows if r['seed']==seed and r['method']=='mature_mtl'])
    paired.append({'seed':seed,'candidate_minus_mtl':float(candidate-baseline)})
output={'prediction_files':count,'cells':len(rows),'max_metric_recompute_error':max(errors),'gradient_error':abs(finite-ga[1,2]),'causal_prefix_errors':causal_errors,'market_ratios':summary,'paired_seed_differences':paired}
(HERE/'VERIFICATION.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))

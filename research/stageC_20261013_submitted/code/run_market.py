"""Run pre-frozen market experiment; all choices use tuning years only."""
from pathlib import Path
import argparse,json,hashlib,time,platform,sys,itertools
import numpy as np
import pandas as pd
from models import *
R=Path(__file__).resolve().parents[1];O=R/'outputs';P=O/'predictions';P.mkdir(exist_ok=True)
def load():
    f=pd.read_csv(R/'raw_private/GCP.csv',parse_dates=['date']);f=f.loc[f.date<'2023-01-01'].copy()
    p=f.price.to_numpy();dates=f.date.to_numpy();first=np.flatnonzero(f.date.dt.year.to_numpy()==2019)[0]
    return make_data(p,dates,int(first))
def configs():
    out=[('RW',{})]+[('drift',{'window':w}) for w in [20,60,120,240]]+[('SES',{'alpha':a}) for a in [.05,.2,.5,.8]]
    out += [('Holt',dict(alpha=a,beta=b,phi=p)) for a,b,p in itertools.product([.05,.2,.5],[.01,.1],[.8,.98])]
    for fam in ['Ridge_diff','Ridge_level']:
        out += [(fam,dict(lag=l,alpha=a)) for l,a in itertools.product([20,60],[.001,.01,.1,1,10,100])]
    out += [('HGBT',dict(leaves=l,l2=a,lr=b)) for l,a,b in itertools.product([7,15,31],[0.,1.],[.05,.1])]
    out += [('SVR',dict(C=c,gamma=g,epsilon=e)) for c,g,e in itertools.product([1.,10.,100.],['scale',.1],[.01,.1])]
    for fam in ['free','conditional','global']:
        out += [(fam,dict(lr=l,wd=w)) for l,w in itertools.product([.001,.003],[.001,.01])]
    return [{'id':f'{fam}_{i:03d}','family':fam,'params':p} for i,(fam,p) in enumerate(out)]
def main(group,limit=None):
    completed=0
    d=load();allc=configs();(O/'configuration_registry.json').write_text(json.dumps(allc,indent=2))
    for c in allc:
        family=c['family'];neural=family in ['free','conditional','global']
        if group=='neural' and not neural:continue
        if group=='baselines' and neural:continue
        if group not in ['neural','baselines','all'] and family!=group:continue
        file=P/f"{c['id']}.npz"
        if file.exists():
            # Operational integrity check added after scoring; no model changes.
            with np.load(file) as cached:
                cached_p=cached['delta'];expected_seeds=3 if neural else 1
                if cached_p.shape!=(expected_seeds,len(d.p),3) or not np.isfinite(cached_p[:,d.first:]).all():
                    raise RuntimeError('Incomplete checkpoint: '+str(file))
            if not (O/f"run_{c['id']}.json").exists():
                raise RuntimeError('Checkpoint missing completion receipt: '+str(file))
            continue
        start=time.perf_counter();logs=[]
        if neural:
            preds=[];taus=[]
            for seed in [11,29,47]:
                p,t,l=neural_forecasts(d,family,**c['params'],seed=seed);preds.append(p);taus.append(t);logs.append(l)
            p=np.array(preds);tau=np.array(taus)
        else:p=(tree_forecasts_batched(d,c['params']) if family=='HGBT' else baseline_forecasts(d,family,c['params']))[None];tau=np.empty(0)
        np.savez_compressed(file,delta=p,tau=tau)
        receipt=dict(**c,seconds=time.perf_counter()-start,logs=logs)
        (O/f"run_{c['id']}.json").write_text(json.dumps(receipt,indent=2))
        print(c['id'],'done',round(receipt['seconds'],2),'s',flush=True)
        completed+=1
        if limit is not None and completed>=limit:break
    print('GROUP COMPLETE',group,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='all');ap.add_argument('--limit',type=int,default=None);a=ap.parse_args();main(a.group,a.limit)

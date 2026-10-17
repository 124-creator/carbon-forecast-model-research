"""Independent score arithmetic, complete saved-weight replay, bounded fresh refits."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import core
from experiment import fit,write_json

def require(x,msg):
    if not x:raise ValueError(msg)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fresh-refits',action='store_true');a=ap.parse_args();out=Path(a.out);core.configure()
    p,dates,receipt=core.load_prefix(a.data);ref=core.reference(p)
    orig=np.arange(459,930);mask=np.array([dates[t][:4]==dates[t+20][:4] for t in orig])
    truth=np.stack([p[t+1:t+21]-p[t] for t in orig]);years=np.array([dates[t][:4] for t in orig])
    require(int(mask.sum())==451,'score support count')
    with np.load(out/'evaluation.npz',allow_pickle=False) as z:
        for k,v in dict(origins=orig,truth=truth,score_mask=mask,years=years).items():require(np.array_equal(v,z[k]),'labels '+k)
    with np.load(out/'reference.npz',allow_pickle=False) as z:
        for k,v in ref.items():require(np.array_equal(v,z[k]),'reference '+k)
    with np.load(out/'predictions.npz',allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
    scores=json.loads((out/'ALL_SCORES.json').read_text());rows=json.loads((out/'FIT_REGISTER.json').read_text())
    require(len(rows)==288 and len(pred)==27,'configuration count')
    unique={(r['mode'],r['seed'],r['block']) for r in rows};require(len(unique)==288,'duplicate fit identities')
    maxdiff=0.;full_epochs=0;count=0
    for r in rows:
        key=f"{r['mode']}_s{r['seed']}_b{r['block']:02}"
        tr=json.loads((out/'logs'/f'{key}.json').read_text())
        require(tr['max_train_target']<tr['first_validation_origin'],'purge')
        require(tr['max_validation_target']<tr['fit_end'],'maturity')
        best=tr['initial_validation'];selected=0
        for e,*_,v,gn in tr['trace']:
            if v<best-1e-6:best=v;selected=e
        require(selected==tr['selected_epoch'],'checkpoint '+key)
        require(np.isfinite(np.array(tr['trace'])).all(),'nonfinite trace');full_epochs+=len(tr['trace'])
        with np.load(out/'fits'/f'{key}.npz',allow_pickle=False) as z:
            model=core.make_model(r['mode'],r['seed'],r['constant_variance'])
            state={k.split('parameter__',1)[1]:torch.as_tensor(z[k].copy()) for k in z.files if k.startswith('parameter__')}
            model.load_state_dict(state)
            pr=core.forecast(model,float(z['scale']),p,ref,z['origins'])
            d=float(np.max(abs(pr-z['predictions'])));maxdiff=max(maxdiff,d)
            require(d<=1e-12,'saved-weight prediction differs '+key)
            si=list(core.SEEDS).index(r['seed']);require(np.array_equal(z['predictions'],pred[r['mode']][si,z['origins']-459]),'assembled forecasts')
        count+=1
    maximum=0.;metric_count=0;fresh_scores={}
    for name,arr in pred.items():
        require(np.isfinite(arr).all(),'nonfinite predictions')
        per_seed=[];ensemble_cells=[];cell_index=0
        for year in ('2020','2021'):
            ii=np.flatnonzero(mask & (years==year))
            for h in (1,5,20):
                actual=truth[ii,h-1];den=float(np.dot(actual,actual)/len(ii))
                vals=[]
                for s in range(len(arr)):
                    e=arr[s,ii,h-1]-actual;vals.append(float(np.dot(e,e)/len(ii)/den))
                per_seed.append(vals)
                e=arr[:,ii,h-1].sum(axis=0)/len(arr)-actual
                ensemble_cells.append(float(np.dot(e,e)/len(ii)/den))
                d=abs(np.mean(vals)-scores[name]['cells'][cell_index]['relative_MSE']);maximum=max(maximum,d)
                require(d<1e-12,'cell score');metric_count+=1;cell_index+=1
        ss=np.mean(per_seed,axis=0);main_score=float(np.mean(ss));ensemble=float(np.mean(ensemble_cells))
        for key,value in [('score',main_score),('ensemble',ensemble)]:
            d=abs(value-scores[name][key]);maximum=max(maximum,d);require(d<1e-12,'aggregate score');metric_count+=1
        fresh_scores[name]=main_score
    # Recompute each OLS combination coefficient independently, using matured origins.
    with np.load(out/'combination_weights.npz',allow_pickle=False) as z:w=z['weights']
    wd=0.
    for si in range(4):
        for i,t in enumerate(orig):
            for h in range(1,21):
                valid=[o for o in orig[:i] if o+h<=t][-120:]
                coef=.5
                if len(valid)>=30:
                    vv=np.array(valid);actual=p[vv+h]-p[vv];g=pred['GRU'][si,vv-459,h-1];b=ref['b'][vv,h-1]
                    diff=b-g;den=float(np.sum(diff*diff))
                    if den>1e-18:coef=float(np.clip(np.sum((actual-g)*diff)/den,0,1))
                wd=max(wd,abs(coef-w[si,i,h-1]))
    require(wd<1e-12,'mature combination weights')
    changes=[]
    if a.fresh_refits:
        q=p.copy();q[459:]+=10000;ref2=core.reference(q)
        for mode in ('INTEGRATED','POINT','FREE_DAMP','CONSTANT','GRU'):
            st=time.perf_counter();m,s,r=fit(p,ref,459,mode,17);n,s2,r2=fit(q,ref2,459,mode,17)
            err=max(float((v-n.state_dict()[k]).abs().max()) for k,v in m.state_dict().items())
            pr1=core.forecast(m,s,p,ref,[458]);pr2=core.forecast(n,s2,q,ref2,[458])
            require(err==0 and s==s2 and np.array_equal(pr1,pr2),'future perturbation '+mode)
            with np.load(out/'fits'/f'{mode}_s17_b00.npz',allow_pickle=False) as z:
                first=max(float(np.max(abs(v.numpy()-z['parameter__'+k]))) for k,v in m.state_dict().items())
            changes.append({'mode':mode,'parameter_future_difference':err,'forecast_future_difference':float(abs(pr1-pr2).max()),
                           'fresh_vs_first_saved_parameter_difference':first,'seconds':time.perf_counter()-st})
    code=json.loads((out/'SOURCE_HASHES.json').read_text());unchanged={}
    for name,sha in code.items():unchanged[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==sha
    require(all(unchanged.values()),'source changed after fit start')
    summary={'passed':True,'neural_fits_checked':count,'epochs_checked':full_epochs,'score_origins':int(mask.sum()),
             'weight_prediction_max_difference':maxdiff,'independent_metrics_recomputed':metric_count,
             'arithmetic_max_difference':maximum,'combination_weight_max_difference':wd,
             'fresh_representative_refits':changes,'source_hashes_unchanged':unchanged,'data':receipt,
             'scope':'Independent arithmetic implementation and same-environment saved-weight/fresh representative replay; not external scientific replication or full second training'}
    write_json(out/'AUDIT.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

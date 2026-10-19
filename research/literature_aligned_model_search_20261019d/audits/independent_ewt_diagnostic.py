#!/usr/bin/env python3
"""Independent feature/fit/selection/score replay with shared ewtpy transform.

Does not import the production diagnostic, pandas, sklearn, StandardScaler or
Ridge. EWT1D itself is explicitly reused and hashed, not independently proved.
This audit is authored after completed results were available. It is not a
preregistration and creates no new scientific model-selection experiment.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path

import numpy as np
from ewtpy import EWT1D

ABS_TOL = 1e-8
REL_TOL = 1e-8
KKT_REL_TOL = 1e-10
MARKETS = ('GUANGZHOU','HUBEI','SHANGHAI_D4','EU_REPORTED')
MODES = ('RAW','SLIDING_PAST_ONLY','FULL_SERIES_RETROSPECTIVE')
GRID = (0.01,0.1,1.0,10.0,100.0,1000.0,10000.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def mse(x):
    return math.fsum(float(v)*float(v) for v in x)/len(x)


def transform(p):
    e,_,_=EWT1D(np.asarray(p,dtype=float),N=3,log=0,detect='locmax',
                completion=1,reg='average',lengthFilter=10,sigmaFilter=5)
    if e.shape!=(len(p),3) or not np.isfinite(e).all():
        raise ValueError('Invalid shared EWT transform')
    return e


def feature(p,t,mode,full=None):
    # Explicit scalar/lag layout, independently rebuilt from the protocol.
    lag_rows=list(range(t-16,t+1))
    if mode=='RAW':
        return np.array([p[i]-p[t] for i in lag_rows],dtype=float)
    e=transform(p[t-63:t+1])[47:64] if mode=='SLIDING_PAST_ONLY' else full[t-16:t+1]
    return np.array([v for k,i in enumerate(lag_rows)
                     for v in (p[i]-p[t],e[k,0]-p[t],e[k,1],e[k,2])],dtype=float)


def replay(args):
    if args.out.exists():
        raise FileExistsError('Existing audit receipt will not be overwritten')
    base=args.study/'results/information_diagnostic'
    files=[base/x for x in ['PROTOCOL_BEFORE_RUN.json','INPUT_MANIFEST.json','METRICS.csv',
                           'SUMMARY.json','FUTURE_PERTURBATION_AUDIT.json']]
    predfile=args.predictions or base/'PREDICTIONS.csv'
    files.append(predfile)
    original_hashes={str(p):sha(p) for p in files}
    protocol=json.loads(files[0].read_text())
    manifest={r['market']:r for r in json.loads(files[1].read_text())}
    saved_metrics={(r['market'],r['mode']):r for r in csv_rows(files[2])}
    saved_summary=json.loads(files[3].read_text())
    saved_perturb={r['market']:r for r in json.loads(files[4].read_text())}
    predictions=csv_rows(predfile)
    saved_pred={(r['market'],r['mode'],int(r['origin_row'])):r for r in predictions}
    failures=[];counts={'scalar_comparisons':0,'reconstructed_configurations':0,'prediction_rows':0}
    maximum={}

    def equal(tag,a,b):
        counts['scalar_comparisons']+=1
        if a!=b:failures.append({'tag':tag,'expected':a,'observed':b})

    def close(tag,a,b):
        counts['scalar_comparisons']+=1
        difference=abs(float(a)-float(b))
        maximum[tag]=max(maximum.get(tag,0.0),difference)
        if not math.isfinite(difference) or difference>ABS_TOL+REL_TOL*abs(float(b)):
            failures.append({'tag':tag,'expected':float(a),'observed':float(b),'absolute_difference':difference})

    equal('source_sha256',protocol['source_sha256'],sha(args.study/'source/literature_diagnostic.py'))
    for k,v in [('window',64),('lookback',17),('modes',3),('horizon_observation_steps',1)]:equal(k,protocol[k],v)
    equal('alpha_grid',protocol['alpha_grid'],list(GRID))
    equal('no_duplicate_prediction_keys',len(saved_pred),len(predictions))
    all_selected=[];trial_rows=[];support=[];perturb=[];max_kkt=0.0
    for market in MARKETS:
        pth=args.repo/'research/cross_market_innovation_20261017/data'/(market+'_development.csv')
        data=sorted(csv_rows(pth),key=lambda r:r['date'])
        dates=[r['date'] for r in data];price=np.array([float(r['price']) for r in data])
        assert np.isfinite(price).all() and len(set(dates))==len(dates)
        assert all(d<'2024-01-01' for d in dates)
        n=len(data);train_boundary=int(.7*n);test_boundary=int(.8*n)
        tr=np.arange(63,train_boundary-1,dtype=int)
        va=np.arange(train_boundary,test_boundary-1,dtype=int)
        te=np.arange(test_boundary,n-1,dtype=int)
        equal('input_sha256',sha(pth),manifest[market]['sha256'])
        equal('input_rows',n,manifest[market]['rows'])
        for role,ii in [('train',tr),('val',va),('test',te)]:equal('count_'+role,len(ii),manifest[market]['counts'][role])
        for key,value in [('start',dates[0]),('end',dates[-1]),
                          ('last_training_label',dates[int(tr[-1])+1]),
                          ('first_validation_origin',dates[int(va[0])]),
                          ('last_validation_label',dates[int(va[-1])+1]),
                          ('first_test_origin',dates[int(te[0])])]:equal(key,value,manifest[market][key])
        assert tr[-1]+1<va[0] and va[-1]+1<te[0]
        support.append({'market':market,'rows':n,'train':len(tr),'validation':len(va),'test':len(te),
                        'max_training_label':int(tr[-1])+1,'first_validation_origin':int(va[0]),
                        'max_validation_label':int(va[-1])+1,'first_test_origin':int(te[0])})
        yt=price[tr+1]-price[tr];yv=price[va+1]-price[va];yz=price[te+1]-price[te]
        full=transform(price)
        poison=price.copy();scale=float(np.std(np.diff(price[:train_boundary])))
        for j,i in enumerate(range(train_boundary+1,n)):
            poison[i]+=scale*(5+math.sin(j/5))
        shared_poison=transform(poison)
        causal=float(np.max(np.abs(feature(price,train_boundary,'SLIDING_PAST_ONLY')-
                                   feature(poison,train_boundary,'SLIDING_PAST_ONLY'))))
        retrospective=float(np.max(np.abs(feature(price,train_boundary,'FULL_SERIES_RETROSPECTIVE',full)-
                                          feature(poison,train_boundary,'FULL_SERIES_RETROSPECTIVE',shared_poison))))
        equal('causal_future_invariance',causal,0.0)
        equal('retrospective_future_dependence',retrospective>1e-8,True)
        close('perturb_causal',causal,saved_perturb[market]['future_perturbation_causal_max_abs'])
        close('perturb_retrospective',retrospective,saved_perturb[market]['future_perturbation_retrospective_max_abs'])
        perturb.append({'market':market,'causal_max_abs':causal,'retrospective_max_abs':retrospective})
        for mode in MODES:
            xx=[np.array([feature(price,int(i),mode,full) for i in ii]) for ii in (tr,va,te)]
            mu=xx[0].mean(axis=0);sd=np.sqrt(((xx[0]-mu)**2).mean(axis=0));sd[sd==0]=1.0
            xtr,xv,xz=[(x-mu)/sd for x in xx]
            xm=xtr.mean(axis=0);ym=float(yt.mean());xc=xtr-xm;yc=yt-ym
            u,s,vh=np.linalg.svd(xc,full_matrices=False)
            uy=u.T@yc;choices=[]
            for alpha in GRID:
                beta=vh.T@((s/(s*s+alpha))*uy)
                intercept=ym-float(xm@beta)
                vmse=mse(xv@beta+intercept-yv)
                kkt=xc.T@(xc@beta-yc)+alpha*beta
                rel=float(np.max(np.abs(kkt))/max(1.0,float(np.max(np.abs(xc.T@yc))),float(alpha*np.max(np.abs(beta)))))
                max_kkt=max(max_kkt,rel)
                if rel>KKT_REL_TOL:failures.append({'tag':'ridge_svd_kkt','market':market,'mode':mode,'alpha':alpha,'relative_residual':rel})
                trial_rows.append({'market':market,'mode':mode,'alpha':alpha,'validation_mse':vmse,'relative_kkt':rel})
                choices.append((vmse,alpha,beta,intercept));counts['reconstructed_configurations']+=1
            vmse,alpha,beta,intercept=min(choices,key=lambda t:(t[0],t[1]))
            pred=xz@beta+intercept;err=pred-yz;test_mse=mse(err);rw=mse(yz)
            saved=saved_metrics[(market,mode)];equal('selected_alpha',alpha,float(saved['alpha']))
            metrics={'validation_mse':vmse,'validation_rw_mse':mse(yv),'test_mse':test_mse,
                     'test_rw_mse':rw,'test_mse_over_rw':test_mse/rw,'test_rmse':math.sqrt(test_mse),
                     'test_mae':math.fsum(abs(float(e)) for e in err)/len(err)}
            for key,value in metrics.items():close(key,value,float(saved[key]))
            equal('test_n',len(te),int(saved['test_n']));equal('tuning_trials',len(GRID),int(saved['tuning_trials']))
            for idx,p,y in zip(te,pred,yz):
                r=saved_pred[(market,mode,int(idx))]
                equal('origin_date',dates[int(idx)],r['origin_date'])
                close('observed_change',y,float(r['observed_change']))
                close('predicted_change',p,float(r['predicted_change']))
                counts['prediction_rows']+=1
            all_selected.append({'market':market,'mode':mode,'alpha':alpha,**metrics})
        print(json.dumps({'market':market,'status':'replayed','current_failures':len(failures)}),flush=True)
    aggregate={mode:math.fsum(r['test_mse_over_rw'] for r in all_selected if r['mode']==mode)/4 for mode in MODES}
    for k,v in aggregate.items():close('aggregate',v,saved_summary['equal_market_mean_mse_over_rw'][k])
    equal('configuration_count',counts['reconstructed_configurations'],84)
    equal('declared_fitted_models',saved_summary['fitted_models'],84)
    equal('selected_count',len(all_selected),saved_summary['selected_forecasters'])
    equal('complete_prediction_rows',counts['prediction_rows'],len(predictions))
    equal('input_artifacts_unchanged',original_hashes,{str(p):sha(p) for p in files})
    core=Path(inspect.getfile(EWT1D))
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'status':'PASS_REPLAY_WITH_SHARED_EWT_TRANSFORM' if not failures else 'FAIL',
            'design_timing':'Written after completed diagnostic results were inspected; tolerances fixed in code before this replay; not preregistration',
            'independent_parts':['CSV data/label/index reconstruction','Explicit feature layout','Train-only population standardization','SVD closed-form Ridge with intercept','All84fixed alpha choices and validation selection','All saved test forecasts and metrics','Four future perturbation comparisons','Equal-market aggregation'],
            'shared_dependency':{'function':'ewtpy.EWT1D','version':importlib.metadata.version('ewtpy'),
                                 'path':str(core),'sha256':sha(core),'mathematical_transform_independently_verified':False},
            'production_imported':False,'sklearn_or_pandas_used':False,'production_results_modified':False,
            'audit_script_sha256':sha(Path(__file__)),'production_source_sha256':protocol['source_sha256'],
            'artifact_sha256':original_hashes,'tolerances':{'absolute':ABS_TOL,'relative':REL_TOL,'ridge_relative_kkt':KKT_REL_TOL},
            'counts':counts,'max_absolute_discrepancy_by_quantity':maximum,'maximum_ridge_relative_kkt':max_kkt,
            'support':support,'future_perturbations':perturb,'trial_validation_scores':trial_rows,
            'selected_metrics':all_selected,'equal_market_mean_mse_over_rw':aggregate,'failures':failures,
            'limitations':['The EWT numerical transform itself is reused, not independently derived.',
                           'Retrospective full-series decomposition deliberately includes future data and is not a deployable forecast.',
                           'All periods are previously exposed development data; this does not reproduce the SWEWT author experiment.',
                           '84 configurations were recomputed by algebra for audit, not84additional scientific experiment fits.',
                           'The production summary fit count follows its completed loop; no separate per-trial execution journal was saved.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'out':str(args.out),'status':result['status'],'counts':counts,
                      'max_prediction_difference':maximum.get('predicted_change'),'max_relative_kkt':max_kkt,'failures':len(failures)},indent=2))
    return 0 if not failures else 1


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--study',type=Path,required=True);ap.add_argument('--predictions',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    raise SystemExit(replay(ap.parse_args()))

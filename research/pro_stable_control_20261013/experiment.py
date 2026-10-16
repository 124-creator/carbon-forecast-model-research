"""One bounded rule: profile-set joint forecast projection.

Run synthetic before market. Do not auto-open validation on a tuning failure.
No use of source row/date-format features; no new market downloads.
"""
from __future__ import annotations
import argparse, json, time, hashlib, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from lowdim import H, filter_bank, profile_nll, bank_parameters, generate, targets, ridge_predict, ratios, features

WINDOWS=[120,240,480]; CS=[0.,2.,6.]
COUNTERS={'qp_calls':0,'qp_failures':0,'vertex_solutions':0,'zero_solutions':0}

def project_hull(M):
    """Euclidean projection of zero onto a finite hull (dimension three).
    Vertex KKT shortcut is exact. SLSQP result checked for simplex feasibility
    and a first-order optimality certificate, never silently accepted.
    """
    M=np.asarray(M,float)
    if M.ndim!=2 or M.shape[1]!=3 or len(M)==0 or not np.isfinite(M).all():raise ValueError('Forecast set')
    v=M[np.argmin(np.sum(M*M,axis=1))]
    vv=float(v@v)
    if vv<1e-24:
        COUNTERS['zero_solutions']+=1;return v.copy()
    if np.min(M@v-vv)>=-1e-10*max(1.,vv):
        COUNTERS['vertex_solutions']+=1;return v.copy()
    COUNTERS['qp_calls']+=1
    k=len(M);x0=np.zeros(k);x0[np.argmin(np.sum(M*M,axis=1))]=1
    res=minimize(lambda w:float((w@M)@(w@M)),x0,jac=lambda w:2*M@(w@M),method='SLSQP',bounds=[(0,1)]*k,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)},options={'ftol':1e-12,'maxiter':200})
    w=res.x; d=w@M; certificate=float(np.min(M@d-d@d))
    if not np.isfinite(d).all() or abs(w.sum()-1)>1e-7 or w.min()<-1e-7 or certificate<-1e-6*max(1.,float(d@d)):
        COUNTERS['qp_failures']+=1;return np.zeros(3)
    return d

def bank_archive(p,start,scale):
    """All rules share the exact same causal state bank and parameter scores."""
    F,ll,states=filter_bank(p); n=len(p); sig=np.sqrt(scale)
    sums=np.cumsum(ll,axis=0)
    out={};diag={}
    for w in WINDOWS:
        names=[f'ML_w{w}']+[f'AVERAGE_w{w}_T{T:g}' for T in [.5,1.,2.]]+[f'SHRINK_w{w}_a{a:g}' for a in [0.,.5,1.]]
        names += [f'JOINT_w{w}_c{c:g}' for c in CS]+[f'RECT_w{w}_c{c:g}' for c in CS]+[f'SETMEAN_w{w}_c{c:g}' for c in CS]
        out.update({name:np.full((n,3),np.nan) for name in names})
        chosen=0; sets={}; logs={}; weights={}
        for t in range(start,n):
            if (t-start)%20==0:
                s=max(1,t-w+1);N=t-s+1
                val=sums[t]-sums[s-1]
                ell=val[:,0]+N*np.log(np.maximum(val[:,1]/N,1e-14))
                chosen=int(np.argmin(ell));de=ell-ell.min()
                sets={c:np.flatnonzero(de<=c+1e-10) for c in CS}
                weights={T:np.exp(-.5*de/T) for T in [.5,1.,2.]}
                weights={T:v/v.sum() for T,v in weights.items()}
            out[f'ML_w{w}'][t]=F[t,chosen]
            for T,bw in weights.items():out[f'AVERAGE_w{w}_T{T:g}'][t]=bw@F[t]
            for a in [0.,.5,1.]:out[f'SHRINK_w{w}_a{a:g}'][t]=a*F[t,chosen]
            for c,idx in sets.items():
                M=F[t,idx]/sig
                out[f'JOINT_w{w}_c{c:g}'][t]=project_hull(M)*sig
                lo=M.min(axis=0);hi=M.max(axis=0)
                out[f'RECT_w{w}_c{c:g}'][t]=np.where(lo>0,lo,np.where(hi<0,hi,0))*sig
                out[f'SETMEAN_w{w}_c{c:g}'][t]=M.mean(axis=0)*sig
                diag[f'{w}:{c}:{t}']={'set_size':len(idx),'rw_in_set':bool(0 in idx),'selected_bank_index':chosen}
    return out,diag

def family(name):return name.split('_')[0]

def select(scores,prefix):
    names=[x for x in scores if family(x)==prefix]
    return min(names,key=lambda x:(scores[x],x))

def synthetic(out):
    cfg=json.loads((Path(__file__).parent/'control_protocol.json').read_text())
    ctrl=json.loads((out/'control_results.json').read_text())
    if not ctrl['gate_passed']:raise RuntimeError('Minimal stable estimator gate failed')
    rows=[];start=time.monotonic()
    for name,par in cfg['processes'].items():
        for seed in cfg['seeds']:
            p=generate(cfg['n'],seed,**par);scale=np.mean(targets(p,np.arange(20,580))**2,axis=0)
            a,d=bank_archive(p,600,scale)
            tune=np.arange(600,980);ev=np.arange(1000,1780)
            tt=targets(p,tune);ee=targets(p,ev)
            sc={k:float(np.mean((v[tune]-tt)**2/scale)) for k,v in a.items()}
            for fam in ['ML','AVERAGE','SHRINK','JOINT','RECT','SETMEAN']:
                k=select(sc,fam);rr=ratios(a[k][ev],ee)
                rows.append(dict(process=name,seed=seed,family=fam,selected=k,tuning_score=sc[k],relative_mse=float(rr.mean()),by_horizon=rr.tolist()))
        print('candidate synthetic completed:',name,flush=True)
    sm={name:{f:float(np.mean([r['relative_mse'] for r in rows if r['process']==name and r['family']==f])) for f in ['ML','AVERAGE','SHRINK','JOINT','RECT','SETMEAN']} for name in cfg['processes']}
    recovery=(1-sm['signal']['JOINT'])/(1-ctrl['summary']['signal']['oracle'])
    passed=(sm['null']['JOINT']<=1.02 and sm['signal']['JOINT']<=.95 and recovery>=.5 and COUNTERS['qp_failures']/max(1,COUNTERS['qp_calls'])<=.001)
    result=dict(summary=sm,oracle_gain_recovered=recovery,gate_passed=bool(passed),rows=rows,solver_counts=COUNTERS.copy(),seconds=time.monotonic()-start)
    (out/'candidate_synthetic.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    return result

def score(p,pred,dates,years):
    vals=[];by=[]
    for year in years:
        idx=np.flatnonzero(dates.year==year)
        origins=idx[idx+20<=idx[-1]]
        y=targets(p,origins);pr=pred[origins]
        if not np.isfinite(pr).all():raise ValueError('Missing predictions')
        rr=ratios(pr,y);vals.append(rr.mean())
        by.append({'year':int(year),'n':len(origins),'relative_mse_by_h':rr.tolist(),'rmse':np.sqrt(np.mean((pr-y)**2,axis=0)).tolist(),'mae':np.mean(abs(pr-y),axis=0).tolist()})
    return float(np.mean(vals)),by

def baseline_archive(p,start):
    out={'RW':np.zeros((len(p),3))};n=len(p)
    for w in WINDOWS:
        pr=np.zeros((n,3))
        for t in range(start,n):pr[t]=H*(p[t]-p[max(0,t-w)])/min(t,w)
        out[f'DRIFT_w{w}']=pr
    for L,a in itertools.product([5,20],[.001,.01,.1,1.]):
        pr=np.full((n,3),np.nan)
        for t in range(start,n,20):
            ix=np.arange(t,min(n,t+20));pr[ix]=ridge_predict(p,t+1,ix,L,a)
        out[f'RIDGE_L{L}_a{a:g}']=pr
    X=features(p,20)
    for leaves,rate,w in itertools.product([7,15],[.05,.1],[240,480]):
        pr=np.full((n,3),np.nan)
        for t in range(start,n,20):
            idx=np.arange(max(20,t-20-w+1),t-20+1)
            model=MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=leaves,learning_rate=rate,l2_regularization=1.,early_stopping=False,random_state=20261013))
            model.fit(X[idx],targets(p,idx))
            ix=np.arange(t,min(n,t+20));pr[ix]=model.predict(X[ix])
        out[f'TREE_l{leaves}_r{rate:g}_w{w}']=pr
        print('tree baseline completed',leaves,rate,w,flush=True)
    return out

def audit_packet(root,out):
    packet=root/'research/pro_data_packet_20261013';m=json.loads((packet/'DATA_PACKET_MANIFEST.json').read_text());rows=[]
    for it in m['files']:
        if it['kind']!='derived_csv':continue
        path=root/it['path'];b=path.read_bytes()
        if hashlib.sha256(b).hexdigest()!=it['sha256']:raise ValueError('Input hash mismatch')
        d=pd.read_csv(path);dates=pd.to_datetime(d.date);v=d.author_reported_price.to_numpy(float)
        if len(d)!=it['rows'] or not np.isfinite(v).all() or dates.duplicated().any() or not dates.is_monotonic_increasing:raise ValueError('Packet validation')
        rows.append(dict(path=it['path'],sha256=it['sha256'],rows=len(d),date_min=str(dates.min().date()),date_max=str(dates.max().date()),weekends=int((dates.dt.weekday>4).sum()),missing=int(np.isnan(v).sum()),same_previous=int(np.sum(np.diff(v)==0)),model_features=['past author_reported_price changes'],excluded_columns=['source_excel_row','source_date_kind']))
    (out/'input_audit.json').write_text(json.dumps(rows,indent=2));return rows

def load_market_prices(root):
    """Hard cutoff: no 2022+ value is returned to a model or scoring function."""
    d=pd.read_csv(Path(root)/'data/parsed/figshare_28682348/guangzhou_reported.csv',usecols=['date','author_reported_price'])
    return d.loc[d.date<='2021-12-31'].reset_index(drop=True)

def market(root,out):
    syn=json.loads((out/'candidate_synthetic.json').read_text())
    if not syn['gate_passed']:
        (out/'market_status.json').write_text(json.dumps({'status':'NOT_RUN_SYNTHETIC_GATE_FAILED','validation_scored':False,'2023_scored':False}));return
    d=load_market_prices(root)
    dates=pd.DatetimeIndex(pd.to_datetime(d.date));p=d.author_reported_price.to_numpy(float)
    start=int(np.flatnonzero(dates.year>=2020)[0]);scale=np.mean(targets(p,np.arange(20,start-20))**2,axis=0)
    a,diag=bank_archive(p,start,scale);a.update(baseline_archive(p,start))
    (out/'market_bank_diagnostics.json').write_text(json.dumps(diag))
    scores={};details={}
    for k,v in a.items():scores[k],details[k]=score(p,v,dates,[2020,2021])
    cand=select(scores,'JOINT')
    non=[k for k in scores if family(k) in ['RW','DRIFT','RIDGE','TREE','ML','AVERAGE','SHRINK']]
    cstar=min(non,key=lambda k:(scores[k],k))
    core=min([k for k in non if family(k) in ['ML','AVERAGE','SHRINK']],key=lambda k:(scores[k],k))
    gains={'RW':1-scores[cand]/scores['RW'],'Cstar':1-scores[cand]/scores[cstar],'core':1-scores[cand]/scores[core]}
    hmean=np.mean([x['relative_mse_by_h'] for x in details[cand]],axis=0)
    passed=(gains['RW']>=.02 and gains['Cstar']>=.02 and gains['core']>=.01 and hmean.max()<=1.02 and COUNTERS['qp_failures']/max(1,COUNTERS['qp_calls'])<=.001)
    rows=[dict(name=k,family=family(k),tuning_relative_mse=scores[k],by_year=details[k]) for k in sorted(scores)]
    np.savez_compressed(out/'market_tuning_predictions.npz',**a)
    pd.DataFrame([dict(name=k,family=family(k),score=v) for k,v in scores.items()]).to_csv(out/'market_tuning_all.csv',index=False)
    result=dict(candidate=cand,Cstar=cstar,core=core,gains=gains,tuning_gate_passed=bool(passed),validation_scored=False,new_history_scored=False,model_input_max_date=str(dates.max().date()),rows=rows,solver_counts=COUNTERS.copy())
    (out/'market_tuning.json').write_text(json.dumps(result,indent=2))
    if not passed:
        (out/'market_status.json').write_text(json.dumps({'status':'STOP_TUNING_GATE_FAILED','validation_scored':False,'2023_scored':False,'hubei_scored':False,'eu_scored':False},indent=2))
    else:
        # Validation requires a separate explicit frozen evaluation after checking
        # all mechanism/identity tests. The entrypoint never opens it automatically.
        (out/'market_status.json').write_text(json.dumps({'status':'TUNING_PASSED_VALIDATION_NOT_AUTOMATIC','validation_scored':False,'2023_scored':False},indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','synthetic','market']);ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('outputs'));arg=ap.parse_args();arg.out.mkdir(exist_ok=True,parents=True)
    if arg.stage=='audit':audit_packet(arg.root,arg.out)
    elif arg.stage=='synthetic':synthetic(arg.out)
    else:market(arg.root,arg.out)

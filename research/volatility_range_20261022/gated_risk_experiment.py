"""Train-only optimized shared-slope logistic gate and static-gate ablation.
Imports feature utilities, never reruns the nine-model experiment.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
from scipy.special import expit, logit
import corrected_risk_experiment as base

ROOT=Path(__file__).resolve().parent
NAMES=['QHAR_gate_dynamic','QHAR_gate_static']
DIAGS=[]; FLAGS=[]; PARAMS=[]

def flag(**item):
    FLAGS.append(item)
    print('FLAG '+json.dumps(item,default=str),flush=True)
    with (ROOT/'gate_optimization_flags.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(item,default=str)+'\n')

def transform_fit(x):
    # Shared own/range center and scale separately for each of three horizons.
    shared=np.concatenate([x[:,:3],x[:,3:6]],axis=0)
    mu=shared.mean(axis=0); sd=shared.std(axis=0); sd=np.where(sd>1e-12,sd,1.)
    dm=x[:,6:].mean(axis=0); ds=x[:,6:].std(axis=0); ds=np.where(ds>1e-12,ds,1.)
    return mu,sd,dm,ds

def transform(x,scaler):
    mu,sd,dm,ds=scaler
    return (x[:,:3]-mu)/sd,(x[:,3:6]-mu)/sd,(x[:,6:]-dm)/ds

def components(theta,own,rng,direct,dynamic):
    b=theta[:7]; aa=theta[7:]
    gdesign=np.column_stack([np.ones(len(own)),direct[:,1:]]) if dynamic else np.ones((len(own),1))
    weight=expit(gdesign@aa)
    mix=own+weight[:,None]*(rng-own)
    z=np.column_stack([np.ones(len(own)),mix,direct])
    return z@b,weight,z,gdesign

def objective(theta,own,rng,direct,y,dynamic):
    eta,w,z,gdesign=components(theta,own,rng,direct,dynamic)
    with np.errstate(over='ignore',invalid='ignore'):
        ratio=y*np.exp(-eta)
        d=(1-ratio)/len(y)
        b=theta[:7]; aa=theta[7:]
        value=np.mean(eta+ratio)+.005*(np.dot(b[1:],b[1:])+np.dot(aa[1:],aa[1:]))
        gb=z.T@d
        ga=gdesign.T@(d*((rng-own)@b[1:4])*w*(1-w))
        gb[1:]+=.01*b[1:]; ga[1:]+=.01*aa[1:]
    return value,np.r_[gb,ga]

def check_gradient():
    random=np.random.default_rng(20261022)
    own=random.normal(size=(31,3)); rng=random.normal(size=(31,3)); direct=random.normal(size=(31,3)); y=random.lognormal(-8,1,size=31)
    errors={}
    for dynamic in [True,False]:
        p=random.normal(0,.1,size=10 if dynamic else 8); p[0]=-8
        args=(own,rng,direct,y,dynamic)
        err=check_grad(lambda q:objective(q,*args)[0],lambda q:objective(q,*args)[1],p)
        assert err<1e-5,err
        errors[str(dynamic)]=float(err)
    return errors

def fit_gate(x,y,dynamic,context):
    scaler=transform_fit(x); own,rng,direct=transform(x,scaler)
    results=[]
    for initial in [.2,.5,.8]:
        theta=np.zeros(10 if dynamic else 8);theta[0]=np.log(y.mean());theta[7]=logit(initial)
        args=(own,rng,direct,y,dynamic)
        opt=minimize(objective,theta,args=args,jac=True,method='L-BFGS-B',options={'maxiter':2500,'ftol':1e-12,'gtol':1e-7,'maxls':50})
        DIAGS.append(dict(context,initial_gate=initial,success=bool(opt.success),status=int(opt.status),message=str(opt.message),nit=int(opt.nit),objective=float(opt.fun),max_abs_gradient=float(np.max(np.abs(opt.jac))),attempt=1))
        if not opt.success or not np.isfinite(opt.fun):
            flag(**context,initial_gate=initial,status=int(opt.status),message=str(opt.message),attempt=1)
            opt=minimize(objective,opt.x if np.isfinite(opt.x).all() else theta,args=args,jac=True,method='BFGS',options={'maxiter':5000,'gtol':1e-6})
            DIAGS.append(dict(context,initial_gate=initial,success=bool(opt.success),status=int(opt.status),message=str(opt.message),nit=int(opt.nit),objective=float(opt.fun),max_abs_gradient=float(np.max(np.abs(opt.jac))),attempt=2))
        if not opt.success or not np.isfinite(opt.fun):
            flag(**context,initial_gate=initial,status=int(opt.status),message=str(opt.message),attempt=2)
        else:
            results.append((float(opt.fun),initial,opt.x.copy()))
    if len(results)!=3:
        raise RuntimeError(f'At least one deterministic start failed: {context}')
    best=min(results,key=lambda item:item[0])
    eta,w,_,_=components(best[2],own,rng,direct,dynamic)
    PARAMS.append(dict(context,n_train=len(y),selected_initial_gate=best[1],objective=best[0],parameters=json.dumps(best[2].tolist()),shared_center=json.dumps(scaler[0].tolist()),shared_sd=json.dumps(scaler[1].tolist()),direct_center=json.dumps(scaler[2].tolist()),direct_sd=json.dumps(scaler[3].tolist()),training_gate_min=float(w.min()),training_gate_max=float(w.max()),training_gate_mean=float(w.mean())))
    return scaler,best[2]

def run_asset(asset,prior):
    d,r,r2,eps,xs,ma,ew=base.feature_data(asset)
    x=xs['QHAR_liquidity']
    finite=np.logical_and.reduce([np.isfinite(xx).all(axis=1) for xx in xs.values()])
    n=len(d);years=d.date.dt.year.to_numpy();origins=np.flatnonzero((d.date>='2024-01-01')&(d.date<'2027-01-01'))
    targets={};eligible={}
    for h in base.HS:
        y=np.full(n,np.nan);same=np.zeros(n,dtype=bool)
        for t in range(n-h):
            y[t]=np.mean(r2[t+1:t+h+1]);same[t]=years[t]==years[t+h]
        targets[h]=y;eligible[h]=same
    fits={};rows=[];weights=[]
    for j,t in enumerate(origins):
        if j%22==0:
            for h in base.HS:
                train=np.flatnonzero(finite & eligible[h] & np.isfinite(targets[h]) & (np.arange(n)+h<=t))
                assert np.max(train+h)<=t
                for name in NAMES:
                    fits[h,name]=fit_gate(x[train],targets[h][train],name.endswith('dynamic'),{'asset':asset,'origin':str(d.date.iloc[t].date()),'origin_index':int(t),'horizon':h,'model':name,'train_origin_max':int(train[-1]),'train_label_end_max':int(train[-1]+h)})
            print(f'GATE {asset} refit {j//22+1} {d.date.iloc[t].date()}',flush=True)
        for h in base.HS:
            old=prior.loc[(asset,t,h)]
            assert np.isclose(old.target,targets[h][t],rtol=1e-12,atol=1e-16,equal_nan=True)
            assert bool(old.scored_common)==bool(eligible[h][t] and np.isfinite(targets[h][t]))
            for name in NAMES:
                scaler,theta=fits[h,name]
                own,rng,direct=transform(x[t:t+1],scaler)
                eta,w,_,_=components(theta,own,rng,direct,name.endswith('dynamic'))
                pred=float(np.exp(eta[0]));assert np.isfinite(pred) and pred>0
                rec=old.to_dict();rec.update(asset=asset,origin_index=int(t),horizon=h,model=name,forecast=pred)
                rec['qlike']=float(np.log(pred)+old.target/pred) if old.scored_common else np.nan
                rec['squared_error']=float((pred-old.target)**2) if old.scored_common else np.nan
                rows.append(rec)
                weights.append({'asset':asset,'origin_index':int(t),'date':old.date,'year':int(years[t]),'horizon':h,'model':name,'gate_weight':float(w[0]),'rel':float(x[t,7]),'flat':float(x[t,8]),'scored_common':bool(old.scored_common)})
    return rows,weights

def main():
    start=time.time();(ROOT/'gate_optimization_flags.jsonl').write_text('',encoding='utf-8')
    grad=check_gradient()
    old=pd.read_csv(ROOT/'predictions_all.csv')
    prior=old[old.model=='MA22'].set_index(['asset','origin_index','horizon'],verify_integrity=True)
    rows=[];weights=[]
    for asset in base.ASSETS:
        rr,ww=run_asset(asset,prior);rows.extend(rr);weights.extend(ww)
    p=pd.DataFrame(rows);p.to_csv(ROOT/'gate_predictions_all.csv',index=False)
    allp=pd.concat([old,p],ignore_index=True)
    common=allp[allp.scored_common].copy()
    assert (common.groupby(['asset','date','horizon']).model.nunique()==11).all()
    assert not allp.duplicated(['asset','date','horizon','model']).any()
    common.to_csv(ROOT/'gate_comparison_predictions_common.csv',index=False)
    summary=[];pair=[]
    for (asset,year,stage,h),group in common.groupby(['asset','year','stage','horizon']):
        loss=group.pivot(index='date',columns='model',values='qlike');mse=group.pivot(index='date',columns='model',values='squared_error')
        assert loss.notna().all().all()
        for model in base.MODELS+NAMES:
            diff=loss[model]-loss.MA22
            summary.append({'asset':asset,'year':year,'stage':stage,'horizon':h,'model':model,'n':len(loss),'qlike':loss[model].mean(),'delta_MA22':diff.mean(),'NW_t_vs_MA22':base.nw_t(diff,max(5,h-1)),'MSE':mse[model].mean(),'MSE_pct_MA22':100*mse[model].mean()/mse.MA22.mean()})
        for model in NAMES:
            for benchmark in base.MODELS+NAMES:
                if benchmark==model:continue
                diff=loss[model]-loss[benchmark]
                pair.append({'asset':asset,'year':year,'stage':stage,'horizon':h,'model':model,'benchmark':benchmark,'n':len(loss),'delta_QLIKE':diff.mean(),'NW_t':base.nw_t(diff,max(5,h-1)),'NW_lag':max(5,h-1),'MSE_ratio_pct':100*mse[model].mean()/mse[benchmark].mean()})
    summary=pd.DataFrame(summary);summary.to_csv(ROOT/'gate_summary.csv',index=False)
    pair=pd.DataFrame(pair);pair.to_csv(ROOT/'gate_pairwise.csv',index=False)
    pd.DataFrame(DIAGS).to_csv(ROOT/'gate_fit_diagnostics.csv',index=False)
    pd.DataFrame(PARAMS).to_csv(ROOT/'gate_parameters.csv',index=False)
    weight=pd.DataFrame(weights);weight.to_csv(ROOT/'gate_weights.csv',index=False)
    quant=weight[weight.scored_common].groupby(['asset','year','horizon','model']).gate_weight.agg(n='size',mean='mean',minimum='min',q05=lambda s:s.quantile(.05),q25=lambda s:s.quantile(.25),median='median',q75=lambda s:s.quantile(.75),q95=lambda s:s.quantile(.95),maximum='max').reset_index()
    quant.to_csv(ROOT/'gate_weight_summary.csv',index=False)
    agg=pair.groupby(['year','horizon','model','benchmark'],as_index=False).agg(delta_QLIKE_equal_asset=('delta_QLIKE','mean'),MSE_ratio_pct_equal_asset=('MSE_ratio_pct','mean'))
    agg.to_csv(ROOT/'gate_pairwise_equal_asset.csv',index=False)
    output=['gate_predictions_all.csv','gate_comparison_predictions_common.csv','gate_summary.csv','gate_pairwise.csv','gate_fit_diagnostics.csv','gate_parameters.csv','gate_weights.csv','gate_weight_summary.csv','gate_pairwise_equal_asset.csv','gate_optimization_flags.jsonl']
    manifest={'runtime_seconds':time.time()-start,'script_sha256':base.digest(Path(__file__)),'base_script_sha256':base.digest(ROOT/'corrected_risk_experiment.py'),'input_base_predictions_sha256':base.digest(ROOT/'predictions_all.csv'),'outputs':{f:base.digest(ROOT/f) for f in output},
      'equation':'log f=b0+sum b_k*((1-w)*own_k+w*range_k)+bg*gap+bq*rel+bf*flat; w=sigmoid(a0+a1*rel+a2*flat); static ablation a1=a2=0 with learned a0',
      'standardization':'training-only own/range share center and sd at each scale, computed from stacked own and range values; gap,rel,flat use separate training centers/sds; standardized rel/flat enter both gate and direct terms',
      'ridge':'mean QLIKE + 0.01/2*(sum b_slopes^2+sum gate_covariate_slopes^2); b0 and a0 unpenalized',
      'optimization':'scipy analytic gradient L-BFGS-B; .2/.5/.8 deterministic initial constant gates, lowest training penalized objective selected; failure flags with BFGS retry; abort if any start still fails',
      'protocol':'exactly original experiment targets, training labels, common scoring origins, cross-year exclusion, horizons and 22-origin refit cadence; no original nine-model retraining',
      'checks':{'analytic_gradient_finite_difference_errors':grad,'all_11_models_same_origins':True,'base_targets_unchanged':True,'base_scoring_flags_identical':True},
      'counts':{'new_prediction_rows':len(p),'all_scored_rows':len(common),'fit_starts':len(DIAGS),'selected_fits':len(PARAMS),'flags':len(FLAGS)},
      'limits':['Exploratory parameter-constrained candidate, no established novelty or untouched test','Gate is predictive mixing weight, not identification of measurement error','Nonconvex optimization; three deterministic starts do not guarantee global optimum','All years already viewed; NW t exploratory, no multiple-testing correction','MSE percentages are ratios times 100; subtract 100 for percent change; QLIKE comparisons use differences only']}
    (ROOT/'gate_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('GATE COMPLETE '+json.dumps(manifest['counts']),flush=True)
    print(agg[(agg.model=='QHAR_gate_dynamic')&agg.benchmark.isin(['MA22','EWMA97','QHAR_gate_static','QHAR_interaction'])].to_string(index=False),flush=True)

if __name__=='__main__':main()

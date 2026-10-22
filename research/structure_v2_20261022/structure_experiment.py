"""Frozen structure-v2 experiment. All writes stay in this directory."""
from pathlib import Path
import sys,json,hashlib,time,traceback,platform,math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import scipy
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'volatility_range_20261022'
sys.path.insert(0,str(OLD))
import corrected_risk_experiment as base
PROTOCOL='7e9e00723c2936b94f6b952538a0d5ef8b4428c95a4523a405dd38024f2ca010'
MODELS=['A2_bounded','A3_free_cross','K1_direct_CC','K2_no_cross','DM_pair','K3_no_DM']
HS=[1,5,22]; DELTA=1e-6
FITS=[]; MULTI=[]; COMP=[]; MIX=[]; EPS=[]; AUDIT=[]; MISSING=[]; PREDS=[]; CURRENT={}; INPUTS={}
OPTIONS={'maxiter':2000,'ftol':1e-12,'gtol':1e-7,'maxls':50}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def csv(name,x):pd.DataFrame(x).to_csv(ROOT/name,index=False,float_format='%.17g',encoding='utf-8-sig')
def check(v,msg):
    if not v:raise RuntimeError(msg+' '+json.dumps(CURRENT,default=str))
def standard(x):
    mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-12,sd,1.)
    return mu,sd,np.column_stack([np.ones(len(x)),(x-mu)/sd])
def design(x,fit):
    x=np.atleast_2d(x);return np.column_stack([np.ones(len(x)),(x-fit['mu'])/fit['sd']])
def positive_objective(b,z,y):
    with np.errstate(over='ignore',invalid='ignore'):
        eta=z@b; ratio=y*np.exp(-eta)
        value=float(np.mean(eta+ratio)+.005*(b[1:]@b[1:]))
        grad=z.T@(1-ratio)/len(y);grad[1:]+=.01*b[1:]
    return value,grad
def fit_positive(x,y,ctx,calibrate):
    global CURRENT
    CURRENT=ctx;mu,sd,z=standard(x);check(len(y)>50 and np.isfinite(x).all() and np.isfinite(y).all() and (y>=0).all() and y.mean()>0,'Invalid positive training rows')
    b=np.zeros(z.shape[1]);b[0]=np.log(y.mean())
    opt=minimize(positive_objective,b,args=(z,y),jac=True,method='L-BFGS-B',options=OPTIONS)
    value,grad=positive_objective(opt.x,z,y);err=float(abs(grad).max())
    scale=float(np.mean(y*np.exp(-z@opt.x))) if calibrate else 1.
    FITS.append(dict(ctx,kind='QLIKE_loglink',success=bool(opt.success),objective=value,max_gradient=err,nit=int(opt.nit),message=str(opt.message),scale_c=scale,mu=mu.tolist(),sd=sd.tolist(),beta=opt.x.tolist()))
    check(opt.success and np.isfinite(value) and np.isfinite(opt.x).all() and err<=1e-4 and np.isfinite(scale) and scale>0,'Positive optimization failed')
    return dict(mu=mu,sd=sd,beta=opt.x.copy(),scale=scale)
def positive_predict(fit,x):
    p=fit['scale']*np.exp(design(x,fit)@fit['beta']);check(np.isfinite(p).all() and (p>0).all(),'Invalid positive forecast');return p

def fit_cross(x,y,ctx):
    global CURRENT
    CURRENT=ctx;mu,sd,z=standard(x);pen=np.eye(z.shape[1])*.01;pen[0,0]=0.
    beta=np.linalg.solve(z.T@z/len(y)+pen,z.T@y/len(y))
    e=z@beta-y;grad=z.T@e/len(y)+pen@beta
    check(np.isfinite(beta).all() and abs(grad).max()<1e-10,'Signed ridge failed')
    FITS.append(dict(ctx,kind='signed_MSE_ridge',success=True,objective=float(.5*np.mean(e*e)+.005*(beta[1:]@beta[1:])),max_gradient=float(abs(grad).max()),mu=mu.tolist(),sd=sd.tolist(),beta=beta.tolist()))
    return dict(mu=mu,sd=sd,beta=beta)

def bounded(z,g,i):
    sg=np.sqrt(g);si=np.sqrt(i);root=sg*si
    raw=2*expit(2*z);q=np.clip(raw,DELTA,2-DELTA)
    f=(sg-si)**2+2*root*q
    rho=q-1
    derivative=8*root*expit(2*z)*expit(-2*z)*((raw>DELTA)&(raw<2-DELTA))
    return f,rho,raw!=q,derivative

def rho_objective(beta,z,y,g,i):
    f,_,_,der=bounded(z@beta,g,i)
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        v=float(np.mean(np.log(f)+y/f)+.005*(beta[1:]@beta[1:]))
        grad=z.T@((1-y/f)*der/f)/len(y);grad[1:]+=.01*beta[1:]
    return v,grad

def fit_rho(x,y,g,i,ctx):
    global CURRENT
    CURRENT=ctx;mu,sd,z=standard(x);solutions=[];records=[]
    for name,start in [('zero',np.zeros(z.shape[1])),('uniform',np.ones(z.shape[1])/z.shape[1])]:
        opt=minimize(rho_objective,start,args=(z,y,g,i),jac=True,method='L-BFGS-B',options=OPTIONS)
        value,grad=rho_objective(opt.x,z,y,g,i);err=float(abs(grad).max())
        rec=dict(ctx,kind='rho_QLIKE',start=name,success=bool(opt.success),objective=value,max_gradient=err,nit=int(opt.nit),message=str(opt.message),mu=mu.tolist(),sd=sd.tolist(),beta=opt.x.tolist())
        FITS.append(rec);records.append(rec)
        check(opt.success and np.isfinite(value) and np.isfinite(opt.x).all() and err<=1e-4,'Rho optimization failed')
        solutions.append(opt)
    pick=int(solutions[1].fun<solutions[0].fun)
    MULTI.append(dict(ctx,objective_zero=records[0]['objective'],objective_uniform=records[1]['objective'],absolute_difference=abs(records[0]['objective']-records[1]['objective']),selected=['zero','uniform'][pick]))
    return dict(mu=mu,sd=sd,beta=solutions[pick].x.copy())

def gradient_checks():
    rng=np.random.default_rng(20261022);x=rng.normal(size=(80,3));_,_,z=standard(x)
    g=np.exp(rng.normal(-8,1,80));i=np.exp(rng.normal(-9,1,80));y=np.exp(rng.normal(-8,1,80));errors=[]
    for beta in [np.zeros(4),np.ones(4)/4,np.array([9.,.01,0.,0.]),np.array([-9.,.01,0.,0.])]:
        analytic=rho_objective(beta,z,y,g,i)[1];numeric=[]
        for j in range(4):
            step=np.zeros(4);step[j]=1e-6
            numeric.append((rho_objective(beta+step,z,y,g,i)[0]-rho_objective(beta-step,z,y,g,i)[0])/2e-6)
        errors.append(float(abs(analytic-numeric).max()))
    check(max(errors)<=1e-5,'Synthetic rho derivative check failed')
    return errors

def rolling(x,eps=None):
    s=pd.Series(x);out=np.column_stack([s.rolling(w,min_periods=w).mean() for w in HS])
    return np.log(out+eps) if eps is not None else out

def targets(x,h):
    a=np.full(len(x),np.nan)
    for t in range(len(x)-h):a[t]=np.mean(x[t+1:t+h+1])
    return a

def context(asset,h,t,d,train,model):
    return dict(asset=asset,horizon=int(h),refit_origin_index=int(t),refit_date=str(d.date.iloc[t].date()),n_train=len(train),train_origin_max=int(train[-1]),train_label_end_max=int(train[-1]+h),model=model)

def load_data():
    source=OLD/'normalized_comparison_predictions_common.csv';old=pd.read_csv(source,float_precision='round_trip')
    check(old.model.nunique()==18,'Expected fixed eighteen saved baselines')
    canonical={(r.asset,int(r.horizon),int(r.origin_index)):r._asdict() for r in old[old.model=='MA22'].itertuples(index=False)}
    check(len(canonical)==5073,'Unexpected scoring sample')
    INPUTS[str(source)]=digest(source);INPUTS[str(Path(base.__file__))]=digest(base.__file__)
    for name in ['online_risk_combination_report.md','online_expanding_transfer_report.md','online_five_report.md']:
        INPUTS[str(OLD/name)]=digest(OLD/name)
    data={}
    for asset in base.ASSETS:
        rawpath=base.DATA/f'{asset}_ohlc.csv';INPUTS[str(rawpath)]=digest(rawpath)
        raw=pd.read_csv(rawpath,parse_dates=['date'])
        bad=~np.isfinite(raw[['open','close']]).all(axis=1)|(raw[['open','close']]<=0).any(axis=1)
        for t in np.flatnonzero(bad):MISSING.append(dict(asset=asset,origin_index=int(t),date=str(raw.date.iloc[t].date()),reason='missing_or_nonpositive_open_close'))
        d,r,r2,eps,xs,ma,ew=base.feature_data(asset)
        oa=d.open.to_numpy()*d.adjclose.to_numpy()/d.close.to_numpy();adj=d.adjclose.to_numpy();previous=np.r_[np.nan,adj[:-1]]
        gap=np.log(oa/previous);intraday=np.log(adj/oa);gap[bad.to_numpy()]=np.nan;intraday[bad.to_numpy()]=np.nan
        valid=np.isfinite(gap)&np.isfinite(intraday)&np.isfinite(r)
        er=float(np.max(abs(gap[valid]+intraday[valid]-np.log(adj[valid]/previous[valid]))))
        es=float(np.max(abs(r2[valid]-gap[valid]**2-intraday[valid]**2-2*gap[valid]*intraday[valid])))
        check(er<=1e-12 and es<=1e-12,'Realized decomposition failed')
        AUDIT.append(dict(asset=asset,native_rows=len(d),epsilon=eps,missing_components=int(bad.sum()),max_gap_id_return_error=er,max_h1_squared_decomposition_error=es))
        years=d.date.dt.year.to_numpy();origins=np.flatnonzero((years>=2024)&(years<=2026))
        values={'gap':gap*gap,'id':intraday*intraday,'cross':2*gap*intraday,'cc':r2}
        features={k:rolling(values[k],eps) for k in ['gap','id','cc']};features['cross']=rolling(gap*intraday)
        labels={h:{k:targets(v,h) for k,v in values.items()} for h in HS}
        eligible={h:np.array([t+h<len(d) and years[t]==years[t+h] for t in range(len(d))]) for h in HS}
        for (a,h,t),row in canonical.items():
            if a!=asset:continue
            np.testing.assert_allclose(labels[h]['cc'][t],row['target'],rtol=1e-10,atol=1e-15)
            check(str(d.date.iloc[t].date())==row['date'] and eligible[h][t],'Native scoring origin mismatch')
        data[asset]=dict(d=d,r=r,r2=r2,eps=eps,xs=xs,years=years,origins=origins,values=values,features=features,labels=labels,eligible=eligible)
    return old,canonical,data

def append_prediction(key,model,pred,canonical,refit):
    check(np.isfinite(pred),'Nonfinite forecast')
    if model!='A3_free_cross':check(pred>0,'Nonpositive scored forecast')
    rec=canonical[key]
    cols=['asset','date','origin_index','year','stage','horizon','target','target_end']
    row={c:rec[c] for c in cols};row.update(model=model,forecast=float(pred),refit_origin_index=int(refit),refit_date=rec['refit_date'],scored_common=True)
    check(int(rec['refit_origin_index'])==int(refit),'Refit schedule differs from original')
    PREDS.append(row)

def run_A(data,canonical):
    global CURRENT
    for asset,obj in data.items():
        d=obj['d'];x=obj['features'];labels=obj['labels'];n=len(d);idx=np.arange(n);state={};refit=-1
        finite=np.logical_and.reduce([np.isfinite(v).all(1) for v in x.values()])
        for j,t in enumerate(obj['origins']):
            if j%22==0:
                refit=t
                for h in HS:
                    all_y=labels[h];valid=np.logical_and.reduce([np.isfinite(v) for v in all_y.values()]);train=np.flatnonzero(finite&valid&obj['eligible'][h]&(idx+h<=t))
                    check(len(train)>50 and np.max(train+h)<=t,'Invalid A training cutoff')
                    fits={}
                    for name in ['gap','id','cc']:fits[name]=fit_positive(x[name][train],all_y[name][train],context(asset,h,t,d,train,'A_'+name),True)
                    fits['cross']=fit_cross(x['cross'][train],all_y['cross'][train],context(asset,h,t,d,train,'A3_cross'))
                    fits['rho']=fit_rho(x['cc'][train],all_y['cc'][train],positive_predict(fits['gap'],x['gap'][train]),positive_predict(fits['id'],x['id'][train]),context(asset,h,t,d,train,'A2_rho'))
                    state[h]=fits
            for h in HS:
                key=(asset,h,int(t))
                if key not in canonical:continue
                CURRENT=dict(asset=asset,horizon=h,origin_index=int(t),phase='A_prediction');check(finite[t],'Missing A scored features')
                fit=state[h];g=float(positive_predict(fit['gap'],x['gap'][t])[0]);i=float(positive_predict(fit['id'],x['id'][t])[0]);cc=float(positive_predict(fit['cc'],x['cc'][t])[0]);cross=float((design(x['cross'][t],fit['cross'])@fit['cross']['beta'])[0])
                z=float((design(x['cc'][t],fit['rho'])@fit['rho']['beta'])[0]);f,rho,clip,_=bounded(np.array([z]),np.array([g]),np.array([i]));a2=float(f[0]);a3=g+i+cross;bound=2*np.sqrt(g*i)
                check(a2>0 and abs(a2-g-i)<=bound*(1+1e-8),'A2 coherence assertion failed')
                COMP.append(dict(asset=asset,horizon=h,origin_index=int(t),date=str(d.date.iloc[t].date()),year=int(obj['years'][t]),G=g,I=i,cross_free=cross,cross_bounded=a2-g-i,rho=float(rho[0]),rho_clipped=bool(clip[0]),rho_saturated=bool(abs(rho[0])>.999),A3_bound_violation=bool(abs(cross)>bound*(1+1e-8)),A3_nonpositive=bool(a3<=0),A2=a2,A3=a3))
                for model,pred in [('A2_bounded',a2),('A3_free_cross',a3),('K1_direct_CC',cc),('K2_no_cross',g+i)]:append_prediction(key,model,pred,canonical,refit)
        print('A completed '+asset,flush=True)
    csv('A_components.csv',COMP);csv('A_multistart.csv',MULTI);csv('A_predictions.csv',PREDS)
    check(len(PREDS)==5073*4,'Incomplete A scoring coverage')

def mixing(data,when,target_asset):
    series={};eps={}
    for asset in ['KRBN','KCCA']:
        obj=data[asset];dates=obj['d'].date;v=obj['r2'];mask=(dates<when).to_numpy()&np.isfinite(v)&(v>0);check(mask.any(),'No positive past squared returns')
        eps[asset]=float(v[mask].mean()*1e-4);check(eps[asset]>0,'Invalid mixing epsilon')
        EPS.append(dict(target_asset=target_asset,refit_date=str(when.date()),source_asset=asset,epsilon=eps[asset],n_positive_past=int(mask.sum()),source_date_max=str(dates[mask].max().date())))
        series[asset]=pd.Series(v+eps[asset],index=dates)
    mix=pd.concat(series,axis=1).dropna();a=mix.KRBN;b=mix.KCCA;geo=np.sqrt(a*b);out=pd.DataFrame(index=mix.index)
    for w in HS:
        ma=a.rolling(w,min_periods=w).mean();mb=b.rolling(w,min_periods=w).mean();mg=geo.rolling(w,min_periods=w).mean();k=mg/np.sqrt(ma*mb)
        ok=k.notna();check((k[ok]>0).all() and (k[ok]<=1+1e-12).all(),'Kappa outside Cauchy bound')
        np.testing.assert_allclose(np.log(mg[ok]),.5*(np.log(ma[ok])+np.log(mb[ok]))+np.log(k[ok]),rtol=1e-12,atol=1e-12)
        if w==1:np.testing.assert_allclose(k[ok],1,rtol=1e-12,atol=1e-12)
        out[f'x{w}']=np.log(mg);out[f'kappa{w}']=k
    return out

def run_B(data,canonical):
    global CURRENT
    for asset,obj in data.items():
        d=obj['d'];idx=np.arange(len(d));liq=obj['xs']['QHAR_liquidity'];state={};refit=-1
        for j,t in enumerate(obj['origins']):
            if j%22==0:
                refit=t;mix=mixing(data,d.date.iloc[t],asset).reindex(pd.DatetimeIndex(d.date));dm=mix[[f'x{h}' for h in HS]].to_numpy();allx=np.column_stack([liq,dm]);finite=np.isfinite(allx).all(1)
                for h in HS:
                    y=obj['labels'][h]['cc'];train=np.flatnonzero(finite&np.isfinite(y)&obj['eligible'][h]&(idx+h<=t));check(len(train)>50 and np.max(train+h)<=t,'Invalid B training cutoff')
                    state[h]={model:fit_positive(xx[train],y[train],context(asset,h,t,d,train,model),False) for model,xx in [('DM_pair',allx),('K3_no_DM',liq)]}
            if any((asset,h,int(t)) in canonical for h in HS):
                for h in HS:MIX.append(dict(asset=asset,date=str(d.date.iloc[t].date()),origin_index=int(t),year=int(obj['years'][t]),window=h,kappa=float(mix[f'kappa{h}'].iloc[t]),members='KRBN|KCCA',n_assets=2,refit_date=str(d.date.iloc[refit].date())))
            for h in HS:
                key=(asset,h,int(t))
                if key not in canonical:continue
                CURRENT=dict(asset=asset,horizon=h,origin_index=int(t),phase='B_prediction');check(finite[t],'Missing B scored features')
                for model,xx in [('DM_pair',allx),('K3_no_DM',liq)]:append_prediction(key,model,float(positive_predict(state[h][model],xx[t])[0]),canonical,refit)
        print('DM_pair completed '+asset,flush=True)

def summarize(allp):
    summary=[];pairs=[]
    for asset,ag in allp.groupby('asset'):
        for h,hg in ag.groupby('horizon'):
            for period in ['2024','2025','2026','2025_2026']:
                g=hg[hg.year.isin([2025,2026])] if period=='2025_2026' else hg[hg.year==int(period)]
                q=g.pivot(index='origin_index',columns='model',values='qlike');se=g.pivot(index='origin_index',columns='model',values='squared_error')
                for name in q:
                    missing=int(q[name].isna().sum());summary.append(dict(asset=asset,period=period,horizon=int(h),model=name,n=len(q),QLIKE=float(q[name].mean()) if missing==0 else np.nan,MSE=float(se[name].mean()),QLIKE_undefined=missing))
                for name in MODELS:
                    for bench in q:
                        if name==bench:continue
                        diff=q[name]-q[bench];missing=int(diff.isna().sum());tie=bool(missing==0 and abs(diff).max()<=1e-12)
                        raw=base.nw_t(diff,max(5,int(h)-1)) if not missing else np.nan;stat=0. if tie else raw
                        pairs.append(dict(asset=asset,period=period,horizon=int(h),model=name,benchmark=bench,n=len(q),QLIKE_undefined=missing,delta_QLIKE=float(diff.mean()) if not missing else np.nan,MSE_ratio=float(se[name].mean()/se[bench].mean()),numerical_tie=tie,NW_t=stat,NW_t_raw=raw,NW_p=math.erfc(abs(stat)/np.sqrt(2)) if np.isfinite(stat) else np.nan))
    pair=pd.DataFrame(pairs);equal=[]
    for key,g in pair.groupby(['period','horizon','model','benchmark']):
        complete=bool(g.delta_QLIKE.notna().all() and len(g)==3)
        equal.append(dict(zip(['period','horizon','model','benchmark'],key),n_assets=len(g),QLIKE_undefined=int(g.QLIKE_undefined.sum()),delta_QLIKE_equal_asset=float(g.delta_QLIKE.mean()) if complete else np.nan,MSE_ratio_equal_asset=float(g.MSE_ratio.mean()),assets_QLIKE_better=int((g.delta_QLIKE<0).sum()),assets_MSE_better=int((g.MSE_ratio<1).sum())))
    return pd.DataFrame(summary),pair,pd.DataFrame(equal)

def diagnostics(data):
    shares=[]
    for asset,obj in data.items():
        for year in sorted(set(obj['years'])):
            ix=(obj['years']==year)&np.logical_and.reduce([np.isfinite(obj['values'][k]) for k in ['gap','id','cross','cc']]);den=float(obj['values']['cc'][ix].sum())
            if not ix.any() or den<=0:continue
            vals={k:float(obj['values'][k][ix].sum()/den) for k in ['gap','id','cross']};check(abs(sum(vals.values())-1)<1e-10,'Annual decomposition shares do not sum to one')
            shares.append(dict(asset=asset,year=int(year),n=int(ix.sum()),**vals))
    csv('annual_decomposition_shares.csv',shares);csv('mix_observations.csv',MIX);csv('mix_epsilon.csv',EPS)
    k=pd.DataFrame(MIX);rows=[]
    for (asset,year,w),g in k.groupby(['asset','year','window']):
        v=g.kappa;rows.append(dict(asset=asset,year=int(year),window=int(w),n=len(v),mean=float(v.mean()),std=float(v.std()),minimum=float(v.min()),q05=float(v.quantile(.05)),median=float(v.median()),q95=float(v.quantile(.95)),maximum=float(v.max())))
    csv('kappa_distribution.csv',rows)

def dump_diagnostics():
    csv('fit_diagnostics.csv',FITS);csv('A_multistart.csv',MULTI);save('algebra_audit.json',AUDIT);save('missing_components.json',MISSING)

def main():
    sys.stdout.reconfigure(encoding='utf-8');start=time.time();check(digest(ROOT/'PROTOCOL_ZH.md')==PROTOCOL,'Protocol not frozen hash')
    check(not (ROOT/'manifest.json').exists(),'Existing experiment manifest: refuse another run')
    manifest=dict(status='started',protocol_sha256=PROTOCOL,script_sha256=digest(__file__),base_script_sha256=digest(base.__file__),models=MODELS,delta=DELTA)
    try:
        errors=gradient_checks();old,canonical,data=load_data();run_A(data,canonical);manifest['A_completed']=True;run_B(data,canonical)
        new=pd.DataFrame(PREDS);check(len(new)==5073*6 and not new.duplicated(['asset','horizon','origin_index','model']).any(),'New model coverage mismatch')
        cols=list(new.columns);old=old[cols].copy();old['target']=[canonical[(r.asset,int(r.horizon),int(r.origin_index))]['target'] for r in old.itertuples(index=False)]
        out=pd.concat([old,new],ignore_index=True);out['qlike_defined']=out.forecast>0;out['qlike']=np.nan;ok=out.qlike_defined;out.loc[ok,'qlike']=np.log(out.loc[ok,'forecast'])+out.loc[ok,'target']/out.loc[ok,'forecast'];out['squared_error']=(out.forecast-out.target)**2
        check(out.groupby(['asset','horizon','origin_index']).model.nunique().eq(24).all(),'Not all twenty-four models share origins')
        summary,pairs,equal=summarize(out)
        for name,frame in [('predictions_common.csv',out),('summary.csv',summary),('pairwise.csv',pairs),('pairwise_equal_asset.csv',equal)]:csv(name,frame)
        diagnostics(data);dump_diagnostics()
        check(INPUTS=={p:digest(p) for p in INPUTS},'An old input/report changed');check(digest(ROOT/'PROTOCOL_ZH.md')==PROTOCOL,'Frozen protocol changed')
        decisions={}
        for model in ['A2_bounded','DM_pair']:
            gates={}
            for h in [5,22]:
                g=equal[(equal.period=='2025')&(equal.horizon==h)&(equal.model==model)&equal.benchmark.isin(['EWMA97','GARCH11_t'])]
                gates[str(h)]=bool(len(g)==2 and (g.delta_QLIKE_equal_asset<0).all() and (g.assets_QLIKE_better>=2).all() and (g.QLIKE_undefined==0).all())
            decisions[model]=dict(by_horizon=gates,retained=all(gates.values()))
        c=pd.DataFrame(COMP);mult=pd.DataFrame(MULTI)
        manifest.update(status='completed',input_hashes=INPUTS,counts=dict(common_origins=5073,new_models=6,new_predictions=len(new),all_models=24,all_predictions=len(out),fits=len(FITS),rho_two_start_fits=len(MULTI),missing_components=len(MISSING),A3_nonpositive=int(c.A3_nonpositive.sum()),A3_bound_violations=int(c.A3_bound_violation.sum()),rho_clipped=int(c.rho_clipped.sum()),rho_saturated=int(c.rho_saturated.sum()),multistart_differences_gt_1e8=int((mult.absolute_difference>1e-8).sum())),checks=dict(gradient_errors=errors,realized_decomposition=all(a['max_gap_id_return_error']<=1e-12 for a in AUDIT),common_origins_identical=True,target_matches_saved_tolerance=True,training_label_end_le_refit=True,training_and_scoring_within_year=True,rho_strict_positive_and_bound=True,all_old_inputs_and_reports_unchanged=True,no_A3_origin_deletion=True,factor_fixed_members=True,kappa_identity_and_bounds=True),max_multistart_objective_difference=float(mult.absolute_difference.max()),decisions_2025=decisions,runtime_seconds=time.time()-start,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
        manifest['outputs']={p.name:digest(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.csv','.json'] and p.name!='manifest.json'}
        save('manifest.json',manifest)
        print(json.dumps(dict(status='completed',counts=manifest['counts'],decisions=decisions),ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        dump_diagnostics();save('failure.json',dict(error=str(exc),context=CURRENT,traceback=traceback.format_exc()))
        manifest.update(status='failed',error=str(exc),context=CURRENT,input_hashes=INPUTS,completed_prediction_count=len(PREDS),fit_count=len(FITS),runtime_seconds=time.time()-start)
        save('manifest.json',manifest)
        (ROOT/'REPORT_ZH.md').write_text('# 结构实验失败记录\n\n固定协议运行出现技术失败，按事前规则停止；不删原点、不回退、不根据结果修改参数。\n\n'+str(exc)+'\n\n详见failure.json与manifest.json。\n',encoding='utf-8')
        raise

if __name__=='__main__':main()

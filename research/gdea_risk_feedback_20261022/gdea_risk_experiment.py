"""GDEA fixed-protocol observed-step variance transfer. Run once after PROTOCOL.md."""
from pathlib import Path
import sys, json, hashlib, time, platform
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
import scipy
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT.parent/'volatility_range_20261022'))
import corrected_risk_experiment as base
SOURCE=REPO/'data/local_only/gdea_close_candidate.csv'
EXPERTS=['EWMA94','EWMA97','HAR_return','QHAR_return','GARCH11_t']
BASE=['MA22']+EXPERTS
K=len(EXPERTS); U=np.ones(K)/K
DIAG=[]; CHECKS={'h1_independent':0,'lb_full_actual':0,'future_component_checks':0}
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def feature_data(asset):
    d=pd.read_csv(SOURCE,parse_dates=['date']); d=d[d.date>='2018-01-01'].reset_index(drop=True)
    assert d.date.is_monotonic_increasing and not d.date.duplicated().any()
    assert np.isfinite(d.reported_end_price).all() and (d.reported_end_price>0).all()
    r=np.log(d.reported_end_price).diff(); r2=r*r
    eps=float(r2[d.date<'2024-01-01'].mean()*1e-4)
    x=np.column_stack([np.log(r2.rolling(w,min_periods=w).mean()+eps) for w in [1,5,22]])
    ma=r2.rolling(22,min_periods=22).mean().to_numpy()
    ew={l:r2.ewm(alpha=1-l,adjust=False,ignore_na=False).mean().to_numpy() for l in [.94,.97]}
    return d,r.to_numpy(),r2.to_numpy(),eps,{'HAR_return':x,'QHAR_return':x},ma,ew

def objective(w,x,y,p):
    u=np.ones(x.shape[1])/x.shape[1]; eta=x@w; ratio=y*np.exp(-eta)
    return float(p@(eta+ratio)+.005*np.sum((w-u)**2)),x.T@(p*(1-ratio))+.01*(w-u)
def fit(x,y,p,context):
    assert len(y)>=44 and np.isfinite(y).all() and (y>=0).all() and np.isfinite(x).all()
    p=np.asarray(p,float); assert (p>0).all(); p=p/p.sum()
    k=x.shape[1]; u=np.ones(k)/k
    opt=minimize(objective,u,args=(x,y,p),jac=True,method='SLSQP',bounds=[(0.,1.)]*k,
                 constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)}],
                 options={'maxiter':1000,'ftol':1e-11,'disp':False})
    v,g=objective(opt.x,x,y,p); gap=float(opt.x@g-g.min())
    rec=dict(context,success=bool(opt.success),n=len(y),objective=v,gap=gap,nit=int(opt.nit),message=str(opt.message))
    DIAG.append(rec)
    if not opt.success or not np.isfinite(opt.x).all() or opt.x.min()<0 or abs(opt.x.sum()-1)>1e-8 or not np.isfinite(v) or gap>1e-5:
        save('online_failure.json',rec); raise RuntimeError(str(rec))
    return opt.x.copy()
def lb(x,origins,t,h,r2,decay):
    lead=np.arange(1,h+1); idx=origins[:,None]+lead
    observed=idx<=t; z=(decay[:,None]*observed).sum(axis=0)
    assert (z>0).all()
    mass=decay[:,None]*observed/(h*z)
    np.testing.assert_allclose(mass.sum(axis=0),np.ones(h)/h,rtol=1e-12,atol=1e-14)
    a=mass.sum(axis=1); np.testing.assert_allclose(a.sum(),1,rtol=1e-12)
    # Read only observed returns: no masked future values or out-of-bounds reads.
    values=np.zeros_like(mass); values[observed]=r2[idx[observed]]
    q=(mass*values).sum(axis=1); assert (a>0).all()
    return q/a,a

def combine(data,d,r2):
    rows=[]; weights=[]; rng=np.random.default_rng(20261022)
    for h,g in data.groupby('horizon'):
        b=g[g.model=='MA22'].sort_values('origin_index').reset_index(drop=True)
        origins=b.origin_index.to_numpy(int); f=g.pivot(index='origin_index',columns='model',values='forecast').loc[origins,EXPERTS].to_numpy(); x=np.log(f)
        yy=np.array([r2[s+1:s+h+1].mean() for s in origins]); np.testing.assert_allclose(yy,b.target,rtol=1e-12,atol=1e-18)
        for i,t in enumerate(origins):
            hist=np.flatnonzero(origins<t); solutions={}
            for scope in ['dynamic','expanding']:
                pool=hist[-252:] if scope=='dynamic' else hist
                matured=pool[origins[pool]+h<=t]
                dec=np.exp2(-(t-origins[pool])/22) if scope=='dynamic' else np.ones(len(pool))
                m=np.minimum(h,t-origins[pool]); assert (m>=1).all()
                known=np.array([r2[s+1:s+int(mm)+1].sum() for s,mm in zip(origins[pool],m)])
                assert (origins[pool]+m<=t).all(); CHECKS['future_component_checks']+=1
                fullmask=origins[pool]+h<=t
                for kind in ['full','partial','LB']:
                    name=kind+'_'+scope
                    selected=matured if kind=='full' else pool
                    context={'horizon':int(h),'origin_index':int(t),'date':b.date.iloc[i],'model':name}
                    if len(matured)<44:
                        w=U.copy(); mode='uniform_warmup'
                    else:
                        if kind=='full': y,p=yy[matured],dec[fullmask]
                        elif kind=='partial': y,p=known/m,dec*m/h
                        else: y,p=lb(x[pool],origins[pool],t,h,r2,dec)
                        w=fit(x[selected],y,p,context); mode='optimized'
                    solutions[name]=w
                    if h==1 and kind!='full':
                        np.testing.assert_allclose(w,solutions['full_'+scope],rtol=1e-8,atol=1e-8)
                        np.testing.assert_allclose(np.exp(x[i]@w),np.exp(x[i]@solutions['full_'+scope]),rtol=1e-9,atol=1e-15)
                        CHECKS['h1_independent']+=1
                    if kind=='LB' and len(matured)>=44 and i%22==0:
                        fy,fa=lb(x[matured],origins[matured],t,h,r2,dec[fullmask])
                        testw=rng.dirichlet(np.ones(K)); fv,fg=objective(testw,x[matured],yy[matured],dec[fullmask]/dec[fullmask].sum()); bv,bg=objective(testw,x[matured],fy,fa/fa.sum())
                        np.testing.assert_allclose(fv,bv,rtol=1e-11,atol=1e-11); np.testing.assert_allclose(fg,bg,rtol=1e-11,atol=1e-11); CHECKS['lb_full_actual']+=1
                    counts=np.minimum(h,t-origins[selected]) if kind!='full' else np.full(len(selected),h)
                    weights.append(dict(context,mode=mode,n_feedback=len(selected),n_mature=len(matured),pool_size=len(pool),train_origin_max=int(origins[selected[-1]]) if len(selected) else -1,observed_end_max=int((origins[selected]+counts).max()) if len(selected) else -1,partial_origins=int((counts<h).sum()),**{e:float(w[j]) for j,e in enumerate(EXPERTS)}))
            for name,w in solutions.items():
                pred=float(np.exp(x[i]@w)); rows.append(dict(asset='GDEA',horizon=int(h),date=b.date.iloc[i],year=int(b.year.iloc[i]),origin_index=int(t),target_end=b.target_end.iloc[i],target=float(b.target.iloc[i]),model=name,forecast=pred))
            for name,pred in [('Equal_geometric',np.exp(x[i]@U)),('Equal_arithmetic',f[i]@U)]:
                rows.append(dict(asset='GDEA',horizon=int(h),date=b.date.iloc[i],year=int(b.year.iloc[i]),origin_index=int(t),target_end=b.target_end.iloc[i],target=float(b.target.iloc[i]),model=name,forecast=float(pred)))
        print('Combined h',h,'origins',len(origins),flush=True)
    return pd.DataFrame(rows),pd.DataFrame(weights)
def summarize(data):
    summaries=[]; pairs=[]
    for h,g in data.groupby('horizon'):
        for period in ['2024','2025','2026','2025_2026']:
            z=g[g.year.isin([2025,2026])] if period=='2025_2026' else g[g.year==int(period)]
            q=z.pivot(index='date',columns='model',values='qlike'); se=z.pivot(index='date',columns='model',values='squared_error')
            for name in q:
                summaries.append(dict(period=period,horizon=int(h),model=name,n=len(q),QLIKE=float(q[name].mean()),MSE=float(se[name].mean()),delta_QLIKE_MA22=float((q[name]-q.MA22).mean()),MSE_ratio_MA22=float(se[name].mean()/se.MA22.mean())))
                for bench in q:
                    if name==bench: continue
                    diff=(q[name]-q[bench]).to_numpy()
                    pairs.append(dict(period=period,horizon=int(h),model=name,benchmark=bench,n=len(q),delta_QLIKE=float(diff.mean()),NW_t=base.nw_t(diff,max(5,int(h)-1)),MSE_ratio=float(se[name].mean()/se[bench].mean())))
    return pd.DataFrame(summaries),pd.DataFrame(pairs)
def main():
    sys.stdout.reconfigure(encoding='utf-8'); start=time.time(); source_hash=digest(SOURCE); assert source_hash=='61b9afaa0784496ff183681bff1c1b49a40bb5b869b91d881756503635476e32'
    assert (ROOT/'PROTOCOL.md').exists(); protocol_hash=digest(ROOT/'PROTOCOL.md')
    d,r,r2,eps,xs,ma,ew=feature_data('GDEA')
    audit=d.assign(previous=d.date.shift(),gap=d.date.diff().dt.days)
    annual=audit.groupby(audit.date.dt.year).agg(n=('date','size'),start=('date','min'),end=('date','max'),max_gap=('gap','max'))
    annual.to_csv(ROOT/'annual_coverage.csv'); audit[audit.gap>14][['previous','date','gap']].to_csv(ROOT/'gaps_gt14.csv',index=False)
    rng=np.random.default_rng(20261022); xx=rng.normal(-8,1,(60,K)); y=rng.lognormal(-8,1,60); p=np.ones(60)/60
    grad_error=float(check_grad(lambda w:objective(w,xx,y,p)[0],lambda w:objective(w,xx,y,p)[1],U)); assert grad_error<1e-4
    # Inject only data adapter/model roster/output root; original implementation bytes unchanged.
    base.ROOT=ROOT; base.MODELS=BASE; base.ASSETS=['GDEA']; base.feature_data=feature_data
    base.STAGES={2024:'warmup_seen',2025:'mechanism_diagnostic_seen',2026:'mechanism_diagnostic_seen'}
    raw=pd.DataFrame(base.run_asset('GDEA')); raw.to_csv(ROOT/'base_predictions_all.csv',index=False,float_format='%.17g')
    pd.DataFrame(base.FITS).to_csv(ROOT/'base_fit_diagnostics.csv',index=False); pd.DataFrame(base.AUDIT).to_csv(ROOT/'sample_audit.csv',index=False)
    save('base_flags.json',base.FLAGS)
    data=raw[raw.scored_common].copy(); combos,weights=combine(data,d,r2)
    cols=['asset','horizon','date','year','origin_index','target_end','target','model','forecast']
    out=pd.concat([data[cols],combos],ignore_index=True); out['stage']=np.where(out.year==2024,'warmup_seen','mechanism_diagnostic_seen')
    out['qlike']=np.log(out.forecast)+out.target/out.forecast; out['squared_error']=(out.target-out.forecast)**2
    keys=['horizon','origin_index']; assert out.groupby(keys).model.nunique().eq(14).all(); assert not out.duplicated(keys+['model']).any()
    assert np.isfinite(out[['target','forecast','qlike','squared_error']]).all().all() and (out.forecast>0).all() and (out.target>=0).all()
    assert (weights.observed_end_max<=weights.origin_index).all(); assert (weights.train_origin_max<weights.origin_index).all()
    summary,pairwise=summarize(out)
    outputs={'predictions_common.csv':out,'weights.csv':weights,'summary.csv':summary,'pairwise.csv':pairwise,'online_fit_diagnostics.csv':pd.DataFrame(DIAG)}
    for name,frame in outputs.items(): frame.to_csv(ROOT/name,index=False,float_format='%.17g',encoding='utf-8-sig')
    assert digest(SOURCE)==source_hash and digest(ROOT/'PROTOCOL.md')==protocol_hash
    manifest={'protocol_sha256':protocol_hash,'script_sha256':digest(__file__),'source':str(SOURCE),'source_sha256':source_hash,'base_script_sha256':digest(Path(base.__file__)),'experts':EXPERTS,'rows':len(d),'epsilon':eps,'counts':{'common_keys':len(out)//14,'predictions':len(out),'base_fits':len(base.FITS),'base_flags':len(base.FLAGS),'online_fits':len(DIAG),'online_failures':sum(not z['success'] for z in DIAG)},'checks':dict(CHECKS,gradient_error=grad_error,input_unchanged=True,max_first_order_gap=max(z['gap'] for z in DIAG)),'outputs':{f:digest(ROOT/f) for f in outputs},'runtime_seconds':time.time()-start,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'interpretation':'Already-seen GDEA endPrice observation-step squared returns; not daily latent volatility, last-trade price or independent confirmation'}
    save('manifest.json',manifest); print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()

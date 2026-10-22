"""Corrected exploratory risk mechanism experiment; run directly with Python.
No probe imports. Forecast origin t observes that asset's complete row t.
"""
from pathlib import Path
import sys, json, hashlib, platform, warnings, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import arch
from arch import arch_model

ROOT = Path(__file__).resolve().parent
DATA = Path(r'E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\22_structural_ohlc_q2_route_20260609\03_raw_data\normalized_yahoo_ohlc')
ASSETS = ['KRBN', 'KEUA', 'KCCA']
HS = [1, 5, 22]
MODELS = ['MA22', 'EWMA94', 'EWMA97', 'GARCH11_t', 'HAR_return', 'QHAR_return', 'QHAR_joint', 'QHAR_liquidity', 'QHAR_interaction']
STAGES = {2024: 'validation_seen', 2025: 'development_seen', 2026: 'diagnostic_seen'}
FLAGS, FITS, INPUTS, AUDIT = [], [], [], []

def flag(**record):
    FLAGS.append(record)
    print('FLAG ' + json.dumps(record, ensure_ascii=False, default=str), flush=True)
    with (ROOT / 'optimization_flags.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def nw_t(d, lag):
    d = np.asarray(d, dtype=float)
    n = len(d)
    z = d - d.mean()
    v = np.dot(z, z) / n
    for k in range(1, min(lag, n-1)+1):
        v += 2 * (1-k/(lag+1)) * np.dot(z[k:], z[:-k]) / n
    return float(d.mean()/np.sqrt(v/n)) if v > 0 else (0.0 if np.all(d == 0) else np.nan)

def feature_data(asset):
    path = DATA / f'{asset}_ohlc.csv'
    d = pd.read_csv(path, parse_dates=['date'])
    assert d.date.is_monotonic_increasing and not d.date.duplicated().any(), asset
    cols = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
    assert np.isfinite(d[cols].to_numpy()).all(), f'{asset}: nonfinite raw data'
    assert (d[cols[:-1]] > 0).all().all() and (d.volume >= 0).all()
    assert (d.high >= d.low).all()
    r = np.log(d.adjclose).diff()
    r2 = r*r
    eps = float(r2[d.date < '2024-01-01'].mean() * 1e-4)
    assert eps > 0
    park = np.log(d.high/d.low)**2 / (4*np.log(2))
    adjusted_open = d.open * d.adjclose/d.close
    gap2 = np.log(adjusted_open/d.adjclose.shift(1))**2
    lr = np.column_stack([np.log(r2.rolling(w, min_periods=w).mean()+eps) for w in [1,5,22]])
    lp = np.column_stack([np.log(park.rolling(w, min_periods=w).mean()+eps) for w in [1,5,22]])
    lg = np.log(gap2.rolling(5, min_periods=5).mean()+eps).to_numpy()[:,None]
    rel = np.tanh(np.log((d.volume+1)/(d.volume.rolling(22, min_periods=22).median()+1))).to_numpy()
    flat = (d.high == d.low).astype(float).to_numpy()
    joint = np.column_stack([lr, lp, lg])
    liquid = np.column_stack([joint, rel, flat])
    xs = {'HAR_return':lr, 'QHAR_return':lr, 'QHAR_joint':joint,
          'QHAR_liquidity':liquid, 'QHAR_interaction':np.column_stack([liquid, rel[:,None]*(lp-lr)])}
    ma = r2.rolling(22, min_periods=22).mean().to_numpy()
    ew = {lam:r2.ewm(alpha=1-lam, adjust=False, ignore_na=False).mean().to_numpy() for lam in [.94,.97]}
    INPUTS.append({'asset':asset, 'path':str(path), 'sha256':digest(path), 'rows':len(d),
                   'start':str(d.date.iloc[0].date()), 'end':str(d.date.iloc[-1].date()), 'epsilon':eps,
                   'zero_returns':int((r2 == 0).sum()), 'flat_rows':int(flat.sum())})
    return d, r.to_numpy(), r2.to_numpy(), eps, xs, ma, ew

def fit_regression(name, x, y, eps, context):
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = np.column_stack([np.ones(len(x)), (x-mu)/sd])
    if name == 'HAR_return':
        ly = np.log(y+eps)
        b = np.linalg.lstsq(z, ly, rcond=None)[0]
        smear = np.exp(ly-z@b).mean()
        return (mu, sd, b, float(smear)), {'success':True, 'method':'OLS_smearing', 'n_train_zero':int((y==0).sum())}
    def objective(b):
        eta = z@b
        with np.errstate(over='ignore', invalid='ignore'):
            ratio = y*np.exp(-eta)
            value = np.mean(eta+ratio) + .005*np.dot(b[1:], b[1:])
            grad = z.T@(1-ratio)/len(y)
            grad[1:] += .01*b[1:]
        return value, grad
    b0 = np.zeros(z.shape[1]); b0[0] = np.log(y.mean())
    opt = minimize(objective, b0, jac=True, method='L-BFGS-B', options={'maxiter':2000, 'ftol':1e-12, 'gtol':1e-7, 'maxls':50})
    if not opt.success or not np.isfinite(opt.fun):
        flag(**context, model=name, attempt=1, status=int(opt.status), message=str(opt.message))
        opt = minimize(objective, opt.x if np.isfinite(opt.x).all() else b0, jac=True, method='BFGS', options={'maxiter':4000, 'gtol':1e-6})
    if not opt.success or not np.isfinite(opt.fun):
        flag(**context, model=name, attempt=2, status=int(opt.status), message=str(opt.message), fatal=True)
        raise RuntimeError(f'Optimization failure: {context} {name}: {opt.message}')
    return (mu, sd, opt.x, None), {'success':True, 'method':'QLIKE_loglink_ridge', 'nit':int(opt.nit), 'objective':float(opt.fun), 'max_abs_gradient':float(np.max(np.abs(opt.jac))), 'n_train_zero':int((y==0).sum())}

def regression_prediction(name, fit, x, eps, context):
    mu, sd, b, smear = fit
    eta = np.r_[1., (x-mu)/sd]@b
    pred = float(np.exp(eta))
    if name == 'HAR_return':
        pred = pred*smear-eps
        if pred <= 0:
            flag(**context, model=name, message='Smearing subtract-epsilon forecast nonpositive; machine-tiny forecast floor applied', raw_forecast=pred)
            pred = np.finfo(float).tiny
    assert np.isfinite(pred) and pred > 0, (context, name, pred)
    return pred

def fit_garch(r, t, context):
    sample = r[1:t+1]*100
    model = arch_model(sample, mean='Zero', vol='GARCH', p=1, o=0, q=1, dist='StudentsT', rescale=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = model.fit(disp='off', show_warning=True, options={'maxiter':2000, 'ftol':1e-9})
    for w in caught:
        flag(**context, model='GARCH11_t', message=str(w.message), category=w.category.__name__)
    if res.convergence_flag:
        flag(**context, model='GARCH11_t', attempt=1, status=int(res.convergence_flag), message=str(res.optimization_result.message))
        res = model.fit(disp='off', show_warning=True, starting_values=res.params, options={'maxiter':10000, 'ftol':1e-8})
    if res.convergence_flag:
        flag(**context, model='GARCH11_t', attempt=2, status=int(res.convergence_flag), fatal=True)
        raise RuntimeError('GARCH optimization failure')
    p = res.params
    v = float(res.conditional_volatility[-1]**2)
    return (float(p['omega']), float(p['alpha[1]']), float(p['beta[1]']), v), {
        'success':True, 'omega':float(p['omega']), 'alpha':float(p['alpha[1]']), 'beta':float(p['beta[1]']), 'nu':float(p['nu']), 'n_train':len(sample)}

def run_asset(asset):
    d,r,r2,eps,xs,ma,ew = feature_data(asset)
    n = len(d); years = d.date.dt.year.to_numpy()
    origins = np.flatnonzero((d.date >= '2024-01-01') & (d.date < '2027-01-01'))
    finite_x = np.logical_and.reduce([np.isfinite(x).all(axis=1) for x in xs.values()])
    assert finite_x[origins].all()
    targets, eligible, endpoints = {}, {}, {}
    for h in HS:
        y = np.full(n,np.nan); end = np.full(n,-1,dtype=int); same = np.zeros(n,dtype=bool)
        for t in range(n-h):
            y[t] = np.mean(r2[t+1:t+h+1])
            end[t] = t+h
            same[t] = years[t] == years[t+h]
        targets[h], eligible[h], endpoints[h] = y, same, end
        for t in origins[origins+h<n]:
            assert y[t] == np.mean(r2[t+1:t+h+1])
    rows=[]; fits={}; refit_index=None; gstate=None
    for j,t in enumerate(origins):
        context={'asset':asset,'origin':str(d.date.iloc[t].date()),'origin_index':int(t)}
        if j % 22 == 0:
            refit_index = t
            gstate,gmeta = fit_garch(r,t,context)
            FITS.append(dict(context,model='GARCH11_t',horizon=0,**gmeta))
            # arch analytical h-step forecasts must equal our recurrence at refit.
            w,a,b,v = gstate
            am = arch_model(r[1:t+1]*100, mean='Zero', vol='GARCH',p=1,q=1,dist='StudentsT',rescale=False)
            fixed = am.fix(np.array([w,a,b,gmeta['nu']]))
            reference = fixed.forecast(horizon=22,method='analytic',reindex=False).variance.iloc[-1].to_numpy()
            vf=w+a*(r[t]*100)**2+b*v; check=[]
            for k in range(22):
                check.append(vf); vf=w+(a+b)*vf
            np.testing.assert_allclose(check, reference, rtol=1e-10,atol=1e-12)
            for h in HS:
                # Labels at s are r[s+1]^2,...,r[s+h]^2, fully observed iff s+h<=t.
                train = np.flatnonzero(finite_x & eligible[h] & np.isfinite(targets[h]) & (np.arange(n)+h <= t))
                assert len(train)>50 and np.max(train+h)<=t
                for name,x in xs.items():
                    fits[h,name],meta = fit_regression(name,x[train],targets[h][train],eps,dict(context,horizon=h))
                    FITS.append(dict(context,model=name,horizon=h,n_train=len(train),train_origin_max=int(train[-1]),train_label_end_max=int(train[-1]+h),**meta))
            print(f'{asset} refit {j//22+1} date={context["origin"]}',flush=True)
        else:
            w,a,b,v = gstate
            gstate=(w,a,b,w+a*(r[t-1]*100)**2+b*v)
        w,a,b,v=gstate
        vf=w+a*(r[t]*100)**2+b*v; path=[]
        for k in range(22):
            path.append(vf/10000); vf=w+(a+b)*vf
        for h in HS:
            preds={'MA22':float(ma[t]), 'EWMA94':float(ew[.94][t]), 'EWMA97':float(ew[.97][t]), 'GARCH11_t':float(np.mean(path[:h]))}
            for name,x in xs.items():
                preds[name]=regression_prediction(name,fits[h,name],x[t],eps,dict(context,horizon=h))
            assert set(preds)==set(MODELS) and all(np.isfinite(vv) and vv>0 for vv in preds.values())
            # Explicit exactly-22-day baseline identity (t included).
            np.testing.assert_allclose(preds['MA22'],np.mean(r2[t-21:t+1]),rtol=1e-12,atol=1e-16)
            y=targets[h][t]
            scored=bool(eligible[h][t] and np.isfinite(y))
            end=int(endpoints[h][t])
            for name,pred in preds.items():
                rows.append({'asset':asset,'date':context['origin'],'origin_index':int(t),'year':int(years[t]),'stage':STAGES[int(years[t])],
                  'horizon':h,'model':name,'target':y,'forecast':pred,'target_end':str(d.date.iloc[end].date()) if end>=0 else '',
                  'refit_origin_index':int(refit_index),'refit_date':str(d.date.iloc[refit_index].date()),'scored_common':scored,
                  'qlike':float(np.log(pred)+y/pred) if scored else np.nan,'squared_error':float((pred-y)**2) if scored else np.nan})
    for h in HS:
        ii=origins
        AUDIT.append({'asset':asset,'horizon':h,'all_origins':len(ii),'scored_origins':int(eligible[h][ii].sum()),
                      'cross_year_removed':int(((endpoints[h][ii]>=0)&~eligible[h][ii]).sum()),'incomplete_future_removed':int((endpoints[h][ii]<0).sum()),
                      'zero_targets_scored':int(((targets[h][ii]==0)&eligible[h][ii]).sum())})
    return rows

def summarize(p):
    summaries=[]
    for (asset,year,stage,h),group in p[p.scored_common].groupby(['asset','year','stage','horizon'],sort=True):
        loss=group.pivot(index='date',columns='model',values='qlike')
        mse=group.pivot(index='date',columns='model',values='squared_error')
        assert list(sorted(loss.columns))==list(sorted(MODELS)) and loss.notna().all().all()
        lag=max(5,h-1)
        for model in MODELS:
            dma=loss[model]-loss.MA22
            dl=loss[model]-loss.QHAR_liquidity
            summaries.append({'asset':asset,'year':year,'stage':stage,'horizon':h,'model':model,'n':len(loss),
              'qlike':float(loss[model].mean()),'delta_MA22':float(dma.mean()),'NW_t_vs_MA22':nw_t(dma,lag),
              'delta_liquidity':float(dl.mean()),'NW_t_vs_liquidity':nw_t(dl,lag),'NW_lag':lag,
              'MSE':float(mse[model].mean()),'MSE_pct_MA22':float(100*mse[model].mean()/mse.MA22.mean()),
              'first_origin':loss.index[0],'last_origin':loss.index[-1]})
    return pd.DataFrame(summaries)

def main():
    start=time.time()
    (ROOT/'optimization_flags.jsonl').write_text('',encoding='utf-8')
    rows=[]
    for asset in ASSETS:
        rows.extend(run_asset(asset))
        pd.DataFrame(rows).to_csv(ROOT/'predictions_all.csv',index=False)
    p=pd.DataFrame(rows)
    scored=p[p.scored_common].copy()
    counts=scored.groupby(['asset','date','horizon']).model.nunique()
    assert (counts==len(MODELS)).all()
    assert not p.duplicated(['asset','date','horizon','model']).any()
    assert (pd.to_datetime(scored.target_end).dt.year.to_numpy()==scored.year.to_numpy()).all()
    assert (scored.target>=0).all()
    scored.to_csv(ROOT/'predictions_scored_common.csv',index=False)
    summary=summarize(p); summary.to_csv(ROOT/'summary.csv',index=False)
    pd.DataFrame(FITS).to_csv(ROOT/'fit_diagnostics.csv',index=False)
    pd.DataFrame(AUDIT).to_csv(ROOT/'sample_audit.csv',index=False)
    # Descriptive equal-cell averaging; no pooled inference across heterogeneous assets.
    aggregate=summary.groupby(['year','stage','horizon','model'],as_index=False).agg(
        assets=('asset','nunique'),delta_MA22_equal_asset=('delta_MA22','mean'),
        delta_liquidity_equal_asset=('delta_liquidity','mean'),MSE_pct_MA22_equal_asset=('MSE_pct_MA22','mean'))
    aggregate.to_csv(ROOT/'summary_equal_asset.csv',index=False)
    outputs=['predictions_all.csv','predictions_scored_common.csv','summary.csv','summary_equal_asset.csv','fit_diagnostics.csv','sample_audit.csv','optimization_flags.jsonl']
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.time()-start,
      'python':sys.executable,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'arch':arch.__version__},
      'script':str(Path(__file__).resolve()),'script_sha256':digest(Path(__file__)), 'inputs':INPUTS,
      'outputs':{f:digest(ROOT/f) for f in outputs},'models':MODELS,'horizons':HS,'stages':STAGES,
      'protocol':{'target':'mean_{k=1..h} log(adjclose[t+k]/adjclose[t+k-1])^2; original zeros retained; no fill or replacement',
        'origin':'asset-specific observed dates from 2024-01-01; row t fully available; no date union',
        'refit':'expanding every 22 asset origins, including origins excluded from scoring; each horizon same refit dates',
        'train':'finite common features; s+h <= refit_t; cross-calendar-year labels excluded from training as well as scoring',
        'MA22':'exact r2[t-21:t+1] mean, includes t',
        'EWMA':'lambda=.94/.97; recursive initial r[1]^2, update including r[t]^2; h-step IGARCH variance constant',
        'GARCH':'arch zero-mean GARCH(1,1) Student-t on return*100; parameters fixed between refits; recursive variance filter updated each day; analytical mean of h future variances; divided by 10000',
        'epsilon':'asset-specific pre2024 mean(r^2)*1e-4, frozen; applied only inside logarithms; targets unmodified',
        'HAR':'OLS log(y+eps), training-residual Duan smearing; exp(xb)*smear-eps; nonpositive forecasts machine-tiny floored with flag',
        'QHAR':'mean(log(f)+y/f) + 0.01/2 sum(standardized slope^2); log link; unpenalized intercept; training-only standardization',
        'features_return':'log(rolling mean r^2 + eps), windows 1,5,22, includes t',
        'features_joint':'return features + log(rolling mean Parkinson + eps), windows 1,5,22 + log(rolling5 adjusted overnight gap^2 + eps)',
        'features_liquidity':'joint + tanh(log((volume+1)/(rolling22median(volume)+1))) + exact high==low indicator',
        'features_interaction':'liquidity + rel*(logmeanParkinson-logmeanr2), three scales',
        'qlike':'log(f)+y/f; y=0 is valid; no log(y)',
        'common_sample':'all 9 methods scored on identical asset/horizon/year origin sets; no model-specific deletion',
        'NW':'Bartlett HAC mean loss difference t statistic, lag max(5,h-1), no finite sample or multiple testing correction; negative favors named model',
        'deltas':'model minus MA22 / model minus QHAR_liquidity; negative better',
        'MSE_pct':'100*MSE_model/MSE_MA22; below 100 better'},
      'limits':['All 2024/2025/2026 stages have been viewed; no clean test, exploratory mechanism probe, no novelty claim',
        'Adjusted-close revision history is unavailable; downloaded adjustment factors are not proven historical point-in-time',
        'Daily squared returns are noisy variance proxies; no intraday realized variance',
        'Flat means high==low, not a zero close-to-close return; gap and Parkinson features are logged',
        'Ridge is averaged-loss convention, 0.01/2 times squared slopes; no hyperparameter search',
        'Cross-year exclusion also applied to training by conservative explicit convention',
        'NW t statistics are exploratory with overlapping labels and no multiplicity adjustment'],
      'checks':{'unique_prediction_keys':True,'all_models_common':True,'target_alignment_direct_slice':True,'MA22_exact22':True,'GARCH_analytic_arch_equivalence_at_refit':True,'training_end_le_refit':True,'no_cross_year_scoring':True},
      'counts':{'prediction_rows_all':len(p),'prediction_rows_scored':len(scored),'summary_rows':len(summary),'flags':len(FLAGS),'fits':len(FITS)}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('COMPLETE '+json.dumps(manifest['counts']),flush=True)
    print(aggregate.to_string(index=False),flush=True)

if __name__=='__main__':
    main()

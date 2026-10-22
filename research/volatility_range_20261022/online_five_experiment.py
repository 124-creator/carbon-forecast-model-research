"""Final fixed ETF expert-pool experiment: four existing experts plus saved GARCH.
Exactly eight configurations; no parameter or expert search.
"""
from pathlib import Path
import sys
import time
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
import online_risk_combination as previous

ROOT = Path(__file__).resolve().parent
EXPERTS = previous.EXPERTS + ['GARCH11_t']
U = np.full(5,.2)
RIDGE = .01
MINIMUM = 44
WINDOW = 252
HALF_LIFE = 22
MODELS = [f'Five_{regime}_{feedback}' for regime in ['expanding','dynamic'] for feedback in ['full','partial','LB']] + ['Five_equal_geometric','Five_equal_arithmetic']
DIAGNOSTICS = []
H1_ERRORS = {regime:{'max_weight_absolute':0.,'max_forecast_relative':0.} for regime in ['expanding','dynamic']}


def objective(w,x,y,a):
    eta = x@w
    ratio = y*np.exp(-eta)
    value = a@(eta+ratio)+RIDGE/2*np.sum((w-U)**2)
    grad = x.T@(a*(1-ratio))+RIDGE*(w-U)
    return float(value),grad


def solve(x,y,a,context):
    a = a/a.sum()
    assert len(y)>=MINIMUM and np.isfinite(x).all() and np.isfinite(y).all() and (y>=0).all()
    opt = minimize(objective,U.copy(),args=(x,y,a),jac=True,method='SLSQP',bounds=[(0.,1.)]*5,
                   constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)}],
                   options={'maxiter':1000,'ftol':1e-11,'disp':False})
    w = opt.x
    value,grad = objective(w,x,y,a)
    gap = float(w@grad-grad.min())
    diag = dict(context,success=bool(opt.success),status=int(opt.status),message=str(opt.message),
                iterations=int(opt.nit),objective=value,simplex_first_order_gap=gap,
                simplex_sum=float(w.sum()),min_weight=float(w.min()),n_feedback=len(y),effective_feedback=float(1/(a@a)))
    DIAGNOSTICS.append(diag)
    if not opt.success or not np.isfinite(w).all() or w.min()<0 or abs(w.sum()-1)>1e-8 or not np.isfinite(value) or gap>1e-5:
        with (ROOT/'online_five_flags.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(diag,ensure_ascii=False)+'\n')
        raise RuntimeError(f'Five-expert failure; no fallback: {diag}')
    return w.copy(),diag


def run_group(asset,h,frame,native):
    meta = frame[frame.model=='Online_expanding'].sort_values('origin_index').reset_index(drop=True)
    origins = meta.origin_index.to_numpy(int)
    dates = pd.to_datetime(meta.date).to_numpy()
    native_dates,r2 = native
    assert np.array_equal(native_dates[origins],dates)
    f = frame.pivot(index='origin_index',columns='model',values='forecast').loc[origins,EXPERTS].to_numpy()
    assert np.isfinite(f).all() and (f>0).all()
    x = np.log(f)
    full_y = np.array([np.mean(r2[s+1:s+h+1]) for s in origins])
    np.testing.assert_allclose(full_y,meta.target,rtol=1e-10,atol=1e-15)
    rows,weights = [],[]
    for i,t in enumerate(origins):
        published = np.flatnonzero(origins<t)
        assert np.array_equal(published,np.arange(i))
        solutions = {}
        for regime in ['expanding','dynamic']:
            shared = published if regime=='expanding' else published[-WINDOW:]
            m = np.minimum(h,t-origins[shared])
            fullmask = origins[shared]+h<=t
            full = shared[fullmask]
            decay = np.ones(len(shared)) if regime=='expanding' else np.exp2(-(t-origins[shared])/HALF_LIFE)
            assert np.all(m>=1) and np.all(origins[shared]+m<=t)
            leads = np.arange(1,h+1)
            observed = leads[None,:]<=m[:,None]
            known = np.zeros((len(shared),h))
            rr,cc = np.nonzero(observed)
            obs_indices = origins[shared][rr]+leads[cc]
            assert np.all(obs_indices<=t) and np.all(native_dates[obs_indices]<=dates[i])
            known[rr,cc] = r2[obs_indices]
            training = {'full':(full,full_y[full],decay[fullmask]),
                        'partial':(shared,known.sum(axis=1)/m,decay*m/h)}
            if len(full)>=MINIMUM:
                z = (decay[:,None]*observed).sum(axis=0)
                assert (z>0).all()
                mass = decay[:,None]*observed/z[None,:]/h
                np.testing.assert_allclose(mass.sum(axis=0),np.full(h,1/h),rtol=1e-12,atol=1e-14)
                a = mass.sum(axis=1)
                np.testing.assert_allclose(a.sum(),1.,rtol=1e-12,atol=1e-14)
                training['LB'] = shared,(mass*known).sum(axis=1)/a,a
            for feedback in ['full','partial','LB']:
                model = f'Five_{regime}_{feedback}'
                selected = full if feedback=='full' else shared
                observed_end = origins[selected]+h if feedback=='full' else origins[selected]+np.minimum(h,t-origins[selected])
                assert np.all(observed_end<=t)
                maxend = int(observed_end.max()) if len(observed_end) else -1
                if len(full)>=MINIMUM:
                    selected,yy,aa = training[feedback]
                    context = dict(asset=asset,horizon=int(h),date=str(pd.Timestamp(dates[i]).date()),origin_index=int(t),
                                   model=model,n_full_feedback=len(full),n_shared_published=len(shared),max_observed_return_index=maxend)
                    w,diag = solve(x[selected],yy,aa,context)
                    mode,effective = 'optimized_mature_feedback',diag['effective_feedback']
                    if feedback=='LB' and np.all(m==h):
                        for tw in [U,solutions[f'Five_{regime}_full'][0]]:
                            fv,fg = objective(tw,x[full],full_y[full],decay/decay.sum())
                            bv,bg = objective(tw,x[shared],yy,aa/aa.sum())
                            np.testing.assert_allclose(fv,bv,rtol=1e-12,atol=1e-12)
                            np.testing.assert_allclose(fg,bg,rtol=1e-10,atol=1e-11)
                else:
                    w,mode,effective = U.copy(),'uniform_less_than_44',0.
                pred = float(np.exp(x[i]@w))
                assert np.isfinite(pred) and pred>0
                solutions[model] = w,pred,mode,effective,len(selected),len(full),len(shared),maxend
                if h==1 and feedback!='full':
                    fw,fp,fmode,*_ = solutions[f'Five_{regime}_full']
                    np.testing.assert_allclose(w,fw,rtol=1e-9,atol=1e-9)
                    np.testing.assert_allclose(pred,fp,rtol=1e-10,atol=1e-15)
                    assert mode==fmode
                    H1_ERRORS[regime]['max_weight_absolute'] = max(H1_ERRORS[regime]['max_weight_absolute'],float(np.max(np.abs(w-fw))))
                    H1_ERRORS[regime]['max_forecast_relative'] = max(H1_ERRORS[regime]['max_forecast_relative'],abs(pred/fp-1))
        solutions['Five_equal_geometric'] = U.copy(),float(np.exp(x[i]@U)),'fixed_equal',0.,0,0,0,-1
        solutions['Five_equal_arithmetic'] = U.copy(),float(np.mean(f[i])),'fixed_equal',0.,0,0,0,-1
        for model in MODELS:
            w,pred,mode,effective,nfeedback,nfull,nshared,maxend = solutions[model]
            row = meta.iloc[i].to_dict()
            row.update(model=model,forecast=pred,qlike=float(np.log(pred)+row['target']/pred),squared_error=float((pred-row['target'])**2))
            rows.append(row)
            weights.append(dict(asset=asset,horizon=int(h),date=row['date'],year=int(row['year']),origin_index=int(t),model=model,
                                mode=mode,n_feedback=nfeedback,n_full_feedback=nfull,n_shared_published=nshared,
                                effective_feedback=effective,max_observed_return_index=maxend,no_future_assertion=True,
                                **{f'w_{ex}':float(w[k]) for k,ex in enumerate(EXPERTS)}))
    print(f'{asset} h={h}: {len(origins)} origins, eight configurations',flush=True)
    return rows,weights


def summarize(allp):
    summaries,pairs = [],[]
    for (asset,h),group in allp.groupby(['asset','horizon']):
        for period in ['2024','2025','2026','2025_2026']:
            g = group[group.year.isin([2025,2026])] if period=='2025_2026' else group[group.year==int(period)]
            loss = g.pivot(index='origin_index',columns='model',values='qlike')
            mse = g.pivot(index='origin_index',columns='model',values='squared_error')
            assert loss.notna().all().all() and mse.notna().all().all()
            for model in loss.columns:
                summaries.append(dict(asset=asset,horizon=h,period=period,model=model,n=len(loss),qlike=loss[model].mean(),MSE=mse[model].mean(),date_start=g.date.min(),date_end=g.date.max()))
            for model in MODELS:
                for benchmark in loss.columns:
                    if model==benchmark:
                        continue
                    d = loss[model]-loss[benchmark]
                    identity = h==1 and any(model in [f'Five_{regime}_{kind}' for kind in ['full','partial','LB']] and benchmark in [f'Five_{regime}_{kind}' for kind in ['full','partial','LB']] for regime in ['expanding','dynamic'])
                    if identity:
                        np.testing.assert_allclose(d,0.,rtol=0.,atol=1e-9)
                    raw_stat = 0. if identity else previous.nw_t(d,max(5,h-1))
                    numerical_tie = bool(np.max(np.abs(d.to_numpy()))<=1e-12)
                    stat = 0. if identity or numerical_tie else raw_stat
                    pairs.append(dict(asset=asset,horizon=h,period=period,model=model,benchmark=benchmark,n=len(loss),delta_QLIKE_negative_better=d.mean(),MSE_ratio=mse[model].mean()/mse[benchmark].mean(),NW_t_negative_better=stat,NW_normal_p_two_sided=math.erfc(abs(stat)/math.sqrt(2)) if np.isfinite(stat) else np.nan,NW_lag=max(5,h-1),h1_mathematical_identity=identity,numerical_tie=numerical_tie,NW_t_raw=raw_stat,NW_normal_p_raw=math.erfc(abs(raw_stat)/math.sqrt(2)) if np.isfinite(raw_stat) else np.nan))
    pairs = pd.DataFrame(pairs)
    aggregates = pairs.groupby(['period','horizon','model','benchmark'],as_index=False).agg(
        delta_QLIKE_equal_asset=('delta_QLIKE_negative_better','mean'),MSE_ratio_equal_asset=('MSE_ratio','mean'),
        assets=('asset','nunique'),assets_QLIKE_better=('delta_QLIKE_negative_better',lambda s:int((s < -1e-12).sum())))
    return pd.DataFrame(summaries),pairs,aggregates


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    started = time.time()
    inputs = [ROOT/'online_expanding_transfer_full27_predictions.csv',ROOT/'online_expanding_transfer_report.md',ROOT/'online_risk_combination_report.md']
    hashes = {p.name:previous.digest(p) for p in inputs}
    old = pd.read_csv(inputs[0],float_precision='round_trip')
    keys = ['asset','horizon','origin_index']
    assert old.groupby(keys).model.nunique().eq(27).all() and old.scored_common.all()
    rng = np.random.default_rng(20261022)
    xt = rng.normal(-8.,.7,size=(60,5));yt = np.exp(rng.normal(-8.,.7,size=60));at = np.full(60,1/60)
    gradient_error = float(check_grad(lambda w:objective(w,xt,yt,at)[0],lambda w:objective(w,xt,yt,at)[1],U))
    assert gradient_error<1e-5,gradient_error
    native = {}
    for asset in old.asset.unique():
        nd,_,r2,_,_,_,_ = previous.base.feature_data(asset)
        native[asset] = nd.date.to_numpy(),r2
    (ROOT/'online_five_flags.jsonl').write_text('',encoding='utf-8')
    rows,weights = [],[]
    for (asset,h),group in old.groupby(['asset','horizon'],sort=False):
        rr,ww = run_group(asset,h,group,native[asset])
        rows.extend(rr);weights.extend(ww)
    predictions = pd.DataFrame(rows);weight = pd.DataFrame(weights)
    assert predictions.groupby(keys).model.nunique().eq(8).all()
    allp = pd.concat([old,predictions],ignore_index=True)
    assert not allp.duplicated(keys+['model']).any()
    assert allp.groupby(keys).model.nunique().eq(35).all()
    assert (weight.max_observed_return_index<=weight.origin_index).all()
    ws = weight.groupby(['asset','year','horizon','model'],as_index=False).agg(n=('date','size'),mean_feedback=('n_feedback','mean'),mean_effective_feedback=('effective_feedback','mean'),**{f'mean_w_{ex}':(f'w_{ex}','mean') for ex in EXPERTS})
    summary,pairs,aggregates = summarize(allp)
    outputs = {'online_five_predictions.csv':predictions,'online_five_full35_predictions.csv':allp,'online_five_weights.csv':weight,
               'online_five_weight_summary.csv':ws,'online_five_summary.csv':summary,'online_five_pairwise.csv':pairs,
               'online_five_equal_asset.csv':aggregates,'online_five_fit_diagnostics.csv':pd.DataFrame(DIAGNOSTICS)}
    for name,frame in outputs.items():
        frame.to_csv(ROOT/name,index=False,float_format='%.17g',encoding='utf-8-sig')
    assert hashes=={p.name:previous.digest(p) for p in inputs},'Old result changed'
    manifest = {'script_sha256':previous.digest(__file__),'plan_sha256':previous.digest(ROOT/'online_five_plan.md'),
                'helper_script_sha256':previous.digest(previous.__file__),'input_hashes':hashes,'experts':EXPERTS,'models':MODELS,
                'parameters':{'ridge':RIDGE,'uniform_weight':.2,'minimum_full_feedback':MINIMUM,'dynamic_shared_published_window':WINDOW,'dynamic_half_life_native_origin_days':HALF_LIFE},
                'feedback':'Full/partial/LB share published origins and startup within each learner; full uses only complete labels, partial uses observed daily components, LB gives each lead mass1/H',
                'checks':{'no_future_observations':True,'same_common_origins_all35':True,'h1_independent_solver_identity':True,'shared_startup':True,'balanced_mass_each_lead_one_over_h':True,'balanced_a_sums_one':True,'all_mature_balanced_objective_gradient_equals_full':True,'old_outputs_unchanged':True},
                'gradient_check_error':gradient_error,'h1_max_errors':H1_ERRORS,'native_inputs':previous.base.INPUTS,
                'counts':{'common_origins':len(predictions)//8,'new_predictions':len(predictions),'all_predictions':len(allp),'optimizations':len(DIAGNOSTICS),'failures':0},
                'max_simplex_first_order_gap':max(d['simplex_first_order_gap'] for d in DIAGNOSTICS),
                'outputs':{name:previous.digest(ROOT/name) for name in outputs},'runtime_seconds':time.time()-started,
                'scope':'Last fixed ETF expert-pool alteration; exactly eight configurations; no pool/temperature/decay search; 2024 warmup and 2025/partial2026 already-seen mechanism diagnostic'}
    (ROOT/'online_five_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    focus = aggregates[aggregates.period.isin(['2025','2026'])&(aggregates.horizon>1)]
    own = focus[((focus.model.isin(['Five_expanding_partial','Five_expanding_LB']))&(focus.benchmark=='Five_expanding_full'))|((focus.model.isin(['Five_dynamic_partial','Five_dynamic_LB']))&(focus.benchmark=='Five_dynamic_full'))]
    print(own.to_string(index=False),flush=True)
    print('FIVE COMPLETE',json.dumps(manifest['counts']),json.dumps(H1_ERRORS),flush=True)


if __name__=='__main__':
    main()

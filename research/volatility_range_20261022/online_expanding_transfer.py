"""Fixed two-candidate transfer test; all prior experiment outputs stay untouched."""
from pathlib import Path
import json
import math
import time
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import online_risk_combination as original

ROOT = Path(__file__).resolve().parent
MODELS = ['PartialFeedback_expanding', 'LeadBalanced_expanding']
EXPERTS = original.EXPERTS
U = original.UNIFORM
RIDGE = original.RIDGE
MINIMUM = original.MIN_FEEDBACK
DIAGNOSTICS = []
H1_ERROR = {'weight_absolute': 0., 'forecast_absolute': 0., 'forecast_relative': 0.}


def solve(x, y, a, context):
    a = a/a.sum()
    assert len(a) >= MINIMUM
    opt = minimize(original.objective, U.copy(), args=(x,y,a), jac=True, method='SLSQP',
                   bounds=[(0.,1.)]*4,
                   constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(4)}],
                   options={'maxiter':1000,'ftol':1e-11,'disp':False})
    w = opt.x
    value, grad = original.objective(w,x,y,a)
    gap = float(w@grad-grad.min())
    record = dict(context, success=bool(opt.success), status=int(opt.status), message=str(opt.message),
                  iterations=int(opt.nit), objective=float(value), simplex_first_order_gap=gap,
                  simplex_sum=float(w.sum()), min_weight=float(w.min()), n_feedback=len(a),
                  effective_feedback=float(1/(a@a)))
    DIAGNOSTICS.append(record)
    if not opt.success or not np.isfinite(w).all() or w.min()<0 or abs(w.sum()-1)>1e-8 or not np.isfinite(value) or gap>1e-5:
        with (ROOT/'online_expanding_transfer_flags.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(record,ensure_ascii=False)+'\n')
        raise RuntimeError(f'Transfer optimization failed; no fallback: {record}')
    return w.copy(), record


def group_predictions(asset,h,frame,native,old_weights):
    meta = frame[frame.model=='Online_expanding'].sort_values('origin_index').reset_index(drop=True)
    origins = meta.origin_index.to_numpy(int)
    dates = pd.to_datetime(meta.date).to_numpy()
    native_dates,r2 = native
    assert np.array_equal(native_dates[origins],dates)
    f = frame.pivot(index='origin_index',columns='model',values='forecast').loc[origins,EXPERTS].to_numpy()
    assert np.isfinite(f).all() and (f>0).all()
    x = np.log(f)
    oldw = old_weights.set_index('origin_index').loc[origins,[f'w_{ex}' for ex in EXPERTS]].to_numpy()
    oldmode = old_weights.set_index('origin_index').loc[origins,'mode'].to_numpy()
    full_y = np.array([np.mean(r2[t+1:t+h+1]) for t in origins])
    np.testing.assert_allclose(full_y,meta.target,rtol=1e-10,atol=1e-15)
    rows,weights = [],[]
    for i,t in enumerate(origins):
        selected = np.flatnonzero(origins<t)
        assert np.array_equal(selected,np.arange(i))
        m = np.minimum(h,t-origins[selected])
        full = selected[origins[selected]+h<=t]
        assert np.all(m>=1) and np.all(origins[selected]+m<=t)
        leads = np.arange(1,h+1)
        mask = leads[None,:] <= m[:,None]
        known = np.zeros((len(selected),h))
        rr,cc = np.nonzero(mask)
        observed_indices = origins[selected][rr]+leads[cc]
        assert np.all(observed_indices<=t)
        assert np.all(native_dates[observed_indices]<=dates[i])
        known[rr,cc] = r2[observed_indices]
        partial_y = known.sum(axis=1)/m
        partial_a = m/h
        inputs = {'PartialFeedback_expanding':(partial_y,partial_a)}
        if len(full)>=MINIMUM:
            z = mask.sum(axis=0)
            assert (z>0).all()
            mass = mask/z[None,:]/h
            np.testing.assert_allclose(mass.sum(axis=0),np.full(h,1/h),rtol=1e-12,atol=1e-14)
            a = mass.sum(axis=1)
            np.testing.assert_allclose(a.sum(),1.,rtol=1e-12,atol=1e-14)
            balanced_y = (mass*known).sum(axis=1)/a
            inputs['LeadBalanced_expanding'] = (balanced_y,a)
            if np.all(m==h):
                for check_w in [U,oldw[i]]:
                    fv,fg = original.objective(check_w,x[selected],full_y[selected],np.full(len(selected),1/len(selected)))
                    bv,bg = original.objective(check_w,x[selected],balanced_y,a/a.sum())
                    np.testing.assert_allclose(bv,fv,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(bg,fg,rtol=1e-10,atol=1e-11)
            assert oldmode[i]=='optimized_mature_feedback'
        else:
            assert oldmode[i]=='uniform_less_than_44'
        for model in MODELS:
            if len(full)>=MINIMUM:
                yy,aa = inputs[model]
                context = dict(asset=asset,horizon=int(h),date=str(pd.Timestamp(dates[i]).date()),
                               origin_index=int(t),model=model,n_full_feedback=len(full),
                               max_observed_return_index=int(observed_indices.max()))
                w,diag = solve(x[selected],yy,aa,context)
                effective = diag['effective_feedback']
                mode = 'optimized_mature_feedback'
            else:
                w,effective,mode = U.copy(),0.,'uniform_less_than_44'
            pred = float(np.exp(x[i]@w))
            assert np.isfinite(pred) and pred>0
            if h==1:
                np.testing.assert_allclose(w,oldw[i],rtol=1e-9,atol=1e-9)
                np.testing.assert_allclose(pred,meta.forecast.iloc[i],rtol=1e-10,atol=1e-15)
                assert mode==oldmode[i]
                H1_ERROR['weight_absolute'] = max(H1_ERROR['weight_absolute'],float(np.max(np.abs(w-oldw[i]))))
                H1_ERROR['forecast_absolute'] = max(H1_ERROR['forecast_absolute'],abs(pred-meta.forecast.iloc[i]))
                H1_ERROR['forecast_relative'] = max(H1_ERROR['forecast_relative'],abs(pred/meta.forecast.iloc[i]-1))
            row = meta.iloc[i].to_dict()
            row.update(model=model,forecast=pred,qlike=float(np.log(pred)+row['target']/pred),
                       squared_error=float((pred-row['target'])**2))
            rows.append(row)
            weights.append(dict(asset=asset,horizon=int(h),date=row['date'],origin_index=int(t),year=int(row['year']),
                                model=model,mode=mode,n_published_origins=len(selected),n_full_feedback=len(full),
                                partial_origins=int(np.sum(m<h)),observed_components=int(m.sum()),
                                max_observed_return_index=int(observed_indices.max()) if len(observed_indices) else -1,
                                effective_feedback=effective,no_future_assertion=True,
                                **{f'w_{ex}':float(w[k]) for k,ex in enumerate(EXPERTS)}))
    print(f'{asset} h={h}: {len(origins)} origins checked',flush=True)
    return rows,weights


def comparisons(allp):
    summaries,pairs = [],[]
    for (asset,h),g in allp.groupby(['asset','horizon']):
        for period in ['2024','2025','2026','2025_2026']:
            frame = g[g.year.isin([2025,2026])] if period=='2025_2026' else g[g.year==int(period)]
            loss = frame.pivot(index='origin_index',columns='model',values='qlike')
            mse = frame.pivot(index='origin_index',columns='model',values='squared_error')
            assert loss.notna().all().all() and mse.notna().all().all()
            for model in loss.columns:
                summaries.append(dict(asset=asset,horizon=h,period=period,model=model,n=len(loss),
                                      qlike=loss[model].mean(),MSE=mse[model].mean(),
                                      date_start=frame.date.min(),date_end=frame.date.max()))
            for model in MODELS:
                for benchmark in loss.columns:
                    if benchmark==model:
                        continue
                    d = loss[model]-loss[benchmark]
                    identical_h1 = h==1 and benchmark in MODELS+['Online_expanding']
                    # h1 numerical roundoff is mathematical identity, not evidence.
                    if identical_h1:
                        np.testing.assert_allclose(d,0.,rtol=0.,atol=1e-9)
                    stat = 0. if identical_h1 else original.nw_t(d,max(5,h-1))
                    pairs.append(dict(asset=asset,horizon=h,period=period,model=model,benchmark=benchmark,n=len(loss),
                                      delta_QLIKE_negative_better=d.mean(),MSE_ratio=mse[model].mean()/mse[benchmark].mean(),
                                      NW_t_negative_better=stat,NW_normal_p_two_sided=math.erfc(abs(stat)/math.sqrt(2)) if np.isfinite(stat) else np.nan,
                                      NW_lag=max(5,h-1),h1_mathematical_identity=identical_h1))
    pairs = pd.DataFrame(pairs)
    aggregates = pairs.groupby(['period','horizon','model','benchmark'],as_index=False).agg(
        delta_QLIKE_equal_asset=('delta_QLIKE_negative_better','mean'),MSE_ratio_equal_asset=('MSE_ratio','mean'),
        assets=('asset','nunique'),assets_QLIKE_better=('delta_QLIKE_negative_better',lambda x:int((x < -1e-12).sum())))
    return pd.DataFrame(summaries),pairs,aggregates


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    started = time.time()
    inputs = [ROOT/'online_full25_predictions_common.csv',ROOT/'online_weights.csv',ROOT/'online_risk_combination_report.md']
    hashes = {p.name:original.digest(p) for p in inputs}
    old = pd.read_csv(inputs[0],float_precision='round_trip')
    oldweights = pd.read_csv(inputs[1],float_precision='round_trip')
    keys = ['asset','horizon','origin_index']
    assert old.groupby(keys).model.nunique().eq(25).all() and old.scored_common.all()
    native = {}
    for asset in old.asset.unique():
        nd,_,r2,_,_,_,_ = original.base.feature_data(asset)
        native[asset] = nd.date.to_numpy(),r2
    (ROOT/'online_expanding_transfer_flags.jsonl').write_text('',encoding='utf-8')
    rows,weights = [],[]
    for (asset,h),group in old.groupby(['asset','horizon'],sort=False):
        ow = oldweights[(oldweights.asset==asset)&(oldweights.horizon==h)&(oldweights.model=='Online_expanding')]
        rr,ww = group_predictions(asset,h,group,native[asset],ow)
        rows.extend(rr);weights.extend(ww)
    predictions = pd.DataFrame(rows)
    weight = pd.DataFrame(weights)
    assert predictions.groupby(keys).model.nunique().eq(2).all()
    assert len(predictions)==len(old)//25*2
    allp = pd.concat([old,predictions],ignore_index=True)
    assert allp.groupby(keys).model.nunique().eq(27).all()
    assert not allp.duplicated(keys+['model']).any()
    assert (weight.max_observed_return_index<=weight.origin_index).all()
    summary,pairs,aggregates = comparisons(allp)
    ws = weight.groupby(['asset','year','horizon','model'],as_index=False).agg(
        n=('date','size'),mean_feedback=('n_published_origins','mean'),mean_effective_feedback=('effective_feedback','mean'),
        **{f'mean_w_{ex}':(f'w_{ex}','mean') for ex in EXPERTS})
    outputs = {'online_expanding_transfer_predictions.csv':predictions,'online_expanding_transfer_full27_predictions.csv':allp,
               'online_expanding_transfer_weights.csv':weight,'online_expanding_transfer_weight_summary.csv':ws,
               'online_expanding_transfer_summary.csv':summary,'online_expanding_transfer_pairwise.csv':pairs,
               'online_expanding_transfer_equal_asset.csv':aggregates,'online_expanding_transfer_fit_diagnostics.csv':pd.DataFrame(DIAGNOSTICS)}
    for name,frame in outputs.items():
        frame.to_csv(ROOT/name,index=False,float_format='%.17g',encoding='utf-8-sig')
    assert hashes=={p.name:original.digest(p) for p in inputs},'Prior output changed'
    manifest = {'script_sha256':original.digest(__file__),'helper_script_sha256':original.digest(original.__file__),
                'plan_sha256':original.digest(ROOT/'online_expanding_transfer_plan.md'),'input_hashes':hashes,
                'models':MODELS,'experts':EXPERTS,'ridge':RIDGE,'minimum_full_feedback':MINIMUM,
                'window':'All scored_common published origins s<t; no truncation, no decay',
                'full_comparator':'Unchanged saved Online_expanding predictions and weights; no refit',
                'partial':'Response knownsum/m, importance m/H; use observed daily r2 only',
                'lead_balanced':'Z_j=count origins with s+j<=t, mass_sj=1/(H*Z_j); a=sum mass, response=sum mass*r2/a',
                'h1_independent_fit_max_errors':H1_ERROR,
                'checks':{'no_future_returns':True,'shared_startup_with_saved_expanding':True,'h1_matches_saved_expanding_tight_tolerance':True,
                          'lead_mass_one_over_h':True,'a_sums_one':True,'all_mature_objective_gradient_equivalence':True,
                          'common_27_model_origins':True,'old_outputs_unchanged':True},
                'counts':{'common_origins':len(predictions)//2,'new_predictions':len(predictions),'all_predictions':len(allp),
                          'optimizations':len(DIAGNOSTICS),'failures':0},
                'native_inputs':original.base.INPUTS,'max_simplex_first_order_gap':max(d['simplex_first_order_gap'] for d in DIAGNOSTICS),
                'outputs':{name:original.digest(ROOT/name) for name in outputs},'runtime_seconds':time.time()-started,
                'evaluation':'2024 warmup; 2025 and partial2026 already seen exploratory mechanism/transfer checks, no novel method claim'}
    (ROOT/'online_expanding_transfer_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    focus = aggregates[aggregates.period.isin(['2025','2026'])&(aggregates.horizon>1)&aggregates.benchmark.isin(['Online_expanding','GARCH11_t','Equal_geometric'])]
    print(focus.to_string(index=False),flush=True)
    print('EXPANDING TRANSFER COMPLETE',json.dumps(manifest['counts']),json.dumps(H1_ERROR),flush=True)


if __name__=='__main__':
    main()

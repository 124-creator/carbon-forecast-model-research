"""Independent original-price scoring, plus explicit representative fresh refits."""
import json,sys,hashlib,importlib.util
import numpy as np
import backbone as m


def main():
    dates,p=m.read_prefix();orig,pred,logs=m.replay(dates,p)
    initial=int(np.sum(dates<'2020-01-01'));cells=[]
    # Compute original price errors (not relying on historical score()).
    for year in ('2020','2021'):
        mask=np.char.startswith(dates[orig],year);oo=orig[mask]
        target=p[oo[:,None]+m.H];forecast=p[oo,None]+pred[mask]
        mse=np.mean((forecast-target)**2,axis=0)
        rw=np.mean((p[oo,None]-target)**2,axis=0)
        cells.append(dict(year=year,n=int(mask.sum()),mse_h=mse.tolist(),rw_mse_h=rw.tolist(),ratio_h=(mse/rw).tolist()))
    score=float(np.mean([r['ratio_h'] for r in cells]))
    if abs(score-0.949793230120762)>1e-12:raise ValueError('historical score mismatch')
    state=m.SOURCE_ROOT/'research/multi_angle_model_search_20261013/state'
    # Saved array comparison is explicitly NOT retraining.
    with np.load(state/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(orig,z['origins'])
        np.testing.assert_allclose(pred,z['MLE480'],rtol=1e-12,atol=1e-12)
        saved_error=float(np.max(np.abs(pred-z['MLE480'])))
    bank,ll,pars=m.base.filter_bank(p)
    # Reconstruct original state.history selection independently, without importing
    # run.py (which mutates historical base.OUT).
    j=0;history=np.zeros((len(p),3))
    for t in range(120,len(p)):
        fit=max(120,initial+40*((t-initial)//40))
        if t==120 or (t-initial)%40==0:
            start=max(1,fit-480);n=fit-start;a=ll[start:fit].sum(0)
            j=int(np.argmin(a[:,0]+n*np.log(np.maximum(a[:,1]/n,1e-14))))
        history[t]=bank[t,j]
    np.testing.assert_array_equal(pred,history[orig])
    future=[]
    # Fresh filter + parameter selection, no cached bank; these are actual refits.
    for fit in (initial,initial+40,initial+200):
        end=min(fit+39,len(p)-21);oo=np.arange(fit,end+1)
        before,log=m.forecast(p,fit,oo)
        changed=p.copy();changed[end+1:]+=10000+np.arange(len(p)-end-1)
        after,other=m.forecast(changed,fit,oo)
        np.testing.assert_array_equal(before,after)
        if log['selected_rho']!=other['selected_rho'] or log['selected_permanent_fraction']!=other['selected_permanent_fraction']:raise ValueError('future refit selection')
        future.append(dict(fit=fit,unchanged_through=end,perturbed_from=end+1,max_error=float(np.max(np.abs(before-after))),fresh_refits=2))
    # Representative stricter first-issuance check: suffix immediately after t.
    oo=np.array([initial]);q=p.copy();q[initial+1:]+=99999
    a,_=m.forecast(p,initial,oo);b,_=m.forecast(q,initial,oo);np.testing.assert_array_equal(a,b)
    missing={k:np.full(len(orig),np.nan) for k in m.COAL_FIELDS}
    fallback=np.full_like(pred,np.nan)
    for fit in range(initial,len(p),40):
        pos=np.where((orig>=fit)&(orig<fit+40))[0]
        if len(pos):fallback[pos],_=m.forecast(p,fit,orig[pos],{k:v[pos] for k,v in missing.items()})
    np.testing.assert_array_equal(pred,fallback)
    mode='normal' if __debug__ else 'optimized'
    source_paths=[m.SOURCE,state/'run.py',state/'verify.py',m.SOURCE_ROOT/'research/local_mature_feedback_20261013/verify.py']
    receipt=dict(mode=mode,passed=True,rows=950,origins=451,initial_rows=initial,score=score,cells=cells,
        historical_saved_max_error=saved_error,history_selection_bitwise=True,missing_fallback_bitwise=True,
        future_price_fresh_refits=future,first_issuance_future_check=True,
        full_grid_retraining=False,fixed_MLE480_recomputed=True,no_pickle=True,no_2022_read=True,
        coal_performance_evaluated=False,source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_paths})
    (m.HERE/f'VERIFICATION_{mode.upper()}.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (m.HERE/'REFIT_PROTOCOL.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    np.savez_compressed(m.HERE/'PRICE_ONLY_LOCAL_ONLY.npz',origins=orig,prediction_changes=pred,dates=dates[orig])
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
